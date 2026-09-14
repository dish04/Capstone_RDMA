#!/bin/bash
# ==============================================================================
# QEMU Cluster Manager for Multi-Node Distributed RDMA / TCP LLM Inference
# Supports 2, 3, 4, or 5 virtual instances
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$WORKSPACE_ROOT/logs"

# Paths to Kernel, Initrd, and Shared Model Weights
# Automatically checks both ~/capstone / ~/qwen_weights and repo paths
if [[ -z "$INITRD" ]]; then
    if [[ -f "$WORKSPACE_ROOT/capstone/initramfs.cpio.gz" ]]; then
        INITRD="$WORKSPACE_ROOT/capstone/initramfs.cpio.gz"
    elif [[ -f "$HOME/capstone/initramfs.cpio.gz" ]]; then
        INITRD="$HOME/capstone/initramfs.cpio.gz"
    fi
fi

if [[ -z "$KERNEL" ]]; then
    if [[ -f "$WORKSPACE_ROOT/capstone/rdma/bzImage" ]]; then
        KERNEL="$WORKSPACE_ROOT/capstone/rdma/bzImage"
    elif [[ -f "$HOME/capstone/rdma/bzImage" ]]; then
        KERNEL="$HOME/capstone/rdma/bzImage"
    fi
fi


if [[ -z "$WEIGHTS_DIR" ]]; then
    if [[ -d "$HOME/qwen_weights" ]]; then
        WEIGHTS_DIR="$HOME/qwen_weights"
    else
        WEIGHTS_DIR="$WORKSPACE_ROOT/qwen_weights"
    fi
fi


BRIDGE="br0"
BRIDGE_IP="192.168.100.1"
NETMASK="24"

mkdir -p "$LOGS_DIR"

usage() {
    echo "Usage: $0 {start <N> [RAM_MB] [CPUS]|stop|status|console <vm_id>}"
    echo "  start <N>     : Launch N virtual instances (N = 2, 3, 4, or 5)"
    echo "                  Optional: RAM_MB (default: 3072), CPUS (default: 2)"
    echo "  stop          : Stop all running VMs and tear down network"
    echo "  status        : Display status of VMs, bridge, and RDMA devices"
    echo "  console <id>  : Connect to serial console of VM <id>"
    exit 1
}

setup_bridge() {
    echo "[NET] Setting up bridge $BRIDGE with IP $BRIDGE_IP/$NETMASK..."
    if ! ip link show "$BRIDGE" >/dev/null 2>&1; then
        sudo ip link add name "$BRIDGE" type bridge
    fi
    sudo ip addr add "$BRIDGE_IP/$NETMASK" dev "$BRIDGE" 2>/dev/null || true
    sudo ip link set "$BRIDGE" up

    # Enable IP forwarding and accept bridge packets
    sudo sysctl -w net.ipv4.ip_forward=1 >/dev/null
    sudo iptables -I INPUT -i "$BRIDGE" -j ACCEPT 2>/dev/null || true
    sudo iptables -I FORWARD -i "$BRIDGE" -j ACCEPT 2>/dev/null || true
}

setup_host_rdma() {
    echo "[RDMA] Setting up host Soft-RoCE device on $BRIDGE..."
    sudo modprobe rdma_rxe 2>/dev/null || true
    sudo modprobe ib_uverbs 2>/dev/null || true
    
    # Check if rxe0 already exists on br0
    if ! rdma link show rxe0 >/dev/null 2>&1; then
        sudo rdma link add rxe0 type rxe netdev "$BRIDGE" 2>/dev/null || true
    fi
}

start_cluster() {
    local num_nodes="$1"
    local ram_mb="${2:-3072}"
    local cpus="${3:-2}"

    if [[ ! "$num_nodes" =~ ^[2-5]$ ]]; then
        echo "Error: Number of nodes must be 2, 3, 4, or 5 (got: '$num_nodes')"
        exit 1
    fi

    echo "================================================="
    echo "Launching Cluster: $num_nodes Nodes | RAM: ${ram_mb}MB each | CPUs: ${cpus} each"
    echo "Kernel : $KERNEL"
    echo "Initrd : $INITRD"
    echo "Weights: $WEIGHTS_DIR"
    echo "================================================="

    # Verify kernel & initrd exist
    if [[ ! -f "$KERNEL" ]]; then
        echo "ERROR: Kernel image not found at $KERNEL"
        exit 1
    fi
    if [[ ! -f "$INITRD" ]]; then
        echo "ERROR: Initramfs not found at $INITRD"
        echo "Running repack_initramfs.sh first..."
        "$SCRIPT_DIR/repack_initramfs.sh"
    fi

    # Clean up any lingering VM instances before launching
    for pid_file in /tmp/qemu_vm*.pid; do
        if [[ -f "$pid_file" ]]; then
            local old_pid=$(sudo cat "$pid_file" 2>/dev/null || cat "$pid_file" 2>/dev/null)
            if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
                sudo kill -9 "$old_pid" 2>/dev/null || true
            fi
            sudo rm -f "$pid_file"
        fi
    done

    setup_bridge
    setup_host_rdma

    # Ensure files are synced to weights folder
    cp -f "$WORKSPACE_ROOT/capstone/rdma/distributed_pipeline.py" "$WEIGHTS_DIR/" 2>/dev/null || true
    cp -f "$WORKSPACE_ROOT/capstone/rdma/vm_worker.py" "$WEIGHTS_DIR/" 2>/dev/null || true
    cp -f "$WORKSPACE_ROOT/capstone/rdma/rdma_pipeline.c" "$WEIGHTS_DIR/" 2>/dev/null || true
    if [[ -f "$WORKSPACE_ROOT/capstone/rdma/rdma_pipeline" ]]; then
        cp -f "$WORKSPACE_ROOT/capstone/rdma/rdma_pipeline" "$WEIGHTS_DIR/" 2>/dev/null || true
    fi

    for ((i=1; i<=num_nodes; i++)); do
        local tap_dev="tap${i}"
        local mac_addr=$(printf "52:54:00:12:34:%02x" "$i")
        local expected_ip="192.168.100.$((i + 1))"
        local pid_file="/tmp/qemu_vm${i}.pid"
        local sock_file="/tmp/vm${i}_console.sock"
        local log_file="$LOGS_DIR/vm${i}.log"

        echo "[VM $i] Preparing interface $tap_dev (MAC: $mac_addr, Target IP: $expected_ip)..."
        sudo ip tuntap add dev "$tap_dev" mode tap 2>/dev/null || true
        sudo ip link set "$tap_dev" master "$BRIDGE" up

        # Remove old socket and log
        rm -f "$sock_file"
        > "$log_file"

        echo "[VM $i] Booting QEMU instance..."
        sudo qemu-system-x86_64 \
            -enable-kvm \
            -cpu host \
            -kernel "$KERNEL" \
            -initrd "$INITRD" \
            -display none \
            -m "${ram_mb}M" \
            -smp "$cpus" \
            -append "console=ttyS0 root=/dev/ram0 rdinit=/init vm_id=${i} total_nodes=${num_nodes}" \
            -netdev tap,id=net0,ifname="$tap_dev",script=no,downscript=no \
            -device virtio-net-pci,netdev=net0,mac="$mac_addr" \
            -fsdev local,security_model=passthrough,id=fsdev0,path="$WEIGHTS_DIR" \
            -device virtio-9p-pci,id=fs0,fsdev=fsdev0,mount_tag=hostshare \
            -chardev socket,id=char0,path="$sock_file",server=on,wait=off,logfile="$log_file" \
            -serial chardev:char0 \
            -pidfile "$pid_file" \
            -daemonize

        sudo chmod 666 "$pid_file" 2>/dev/null || true
        local pid=$(sudo cat "$pid_file" 2>/dev/null || cat "$pid_file" 2>/dev/null || echo 'Unknown')
        echo "[VM $i] Started (PID: $pid)"
    done

    echo "Waiting for VMs to boot and configure network..."
    for ((i=1; i<=num_nodes; i++)); do
        local target_ip="192.168.100.$((i + 1))"
        local online=0
        echo -n "  -> Probing VM $i ($target_ip)... "
        for attempt in {1..20}; do
            if ping -c 1 -W 1 "$target_ip" >/dev/null 2>&1; then
                online=1
                break
            fi
            sleep 1
        done
        if [[ $online -eq 1 ]]; then
            echo "[✓] ONLINE"
        else
            echo "[!] Did not respond yet (check $LOGS_DIR/vm${i}.log)"
        fi
    done

    echo "Ensuring VM Worker Services are active..."
    for ((i=1; i<=num_nodes; i++)); do
        local sock_file="/tmp/vm${i}_console.sock"
        if [[ -S "$sock_file" ]]; then
            printf "\npython3 /mnt/weights/vm_worker.py --vm-id %d --port 18000 >/tmp/vm_worker.log 2>&1 &\n" "$i" | nc -U "$sock_file" 2>/dev/null || true
        fi
    done

    echo "================================================="
    echo "Cluster start command finished."
}

stop_cluster() {
    echo "Stopping all QEMU cluster instances..."
    for pid_file in /tmp/qemu_vm*.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(sudo cat "$pid_file" 2>/dev/null || cat "$pid_file" 2>/dev/null)
            if [[ -n "$pid" ]]; then
                echo "Terminating VM PID $pid..."
                sudo kill -9 "$pid" 2>/dev/null || true
            fi
            sudo rm -f "$pid_file"
        fi
    done

    sudo killall -9 qemu-system-x86_64 2>/dev/null || true

    echo "Cleaning up TAP devices..."
    for ((i=1; i<=10; i++)); do
        if ip link show "tap${i}" >/dev/null 2>&1; then
            sudo ip link set "tap${i}" down 2>/dev/null || true
            sudo ip link delete "tap${i}" 2>/dev/null || true
        fi
    done

    echo "Cleaning up host RDMA link and bridge..."
    sudo rdma link del rxe0 2>/dev/null || true
    if ip link show "$BRIDGE" >/dev/null 2>&1; then
        sudo ip link set "$BRIDGE" down 2>/dev/null || true
        sudo ip link delete "$BRIDGE" 2>/dev/null || true
    fi

    # Clean up shared memory mailboxes
    sudo rm -f /dev/shm/pipeline_* /dev/shm/llm_buffer* /tmp/vm*_console.sock 2>/dev/null || true
    echo "Cluster stopped cleanly."
}

status_cluster() {
    echo "================ CLUSTER STATUS ================"
    echo "1. Bridge Interface ($BRIDGE):"
    if ip link show "$BRIDGE" >/dev/null 2>&1; then
        ip -br addr show "$BRIDGE"
    else
        echo "   $BRIDGE is DOWN/NOT CONFIGURED"
    fi

    echo "2. RDMA Device (rxe0):"
    if which rdma >/dev/null 2>&1; then
        rdma link show 2>/dev/null || echo "   No RDMA links"
    fi

    echo "3. Active VM Processes:"
    local running=0
    for pid_file in /tmp/qemu_vm*.pid; do
        if [[ -f "$pid_file" ]]; then
            local id=$(basename "$pid_file" | tr -dc '0-9')
            local pid=$(sudo cat "$pid_file" 2>/dev/null || cat "$pid_file" 2>/dev/null)
            if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
                local ip="192.168.100.$((id + 1))"
                local ping_res="OFFLINE"
                if ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then
                    ping_res="ONLINE"
                fi
                echo "   VM $id: PID $pid | IP: $ip ($ping_res) | Console: /tmp/vm${id}_console.sock"
                running=$((running + 1))
            fi
        fi
    done

    if [[ $running -eq 0 ]]; then
        echo "   No active VM instances."
    fi
    echo "================================================"
}

connect_console() {
    local vm_id="$1"
    local sock_file="/tmp/vm${vm_id}_console.sock"
    if [[ ! -S "$sock_file" ]]; then
        echo "Error: Console socket $sock_file not found. Is VM $vm_id running?"
        exit 1
    fi
    echo "Connecting to VM $vm_id serial console (Ctrl+C to exit)..."
    if which nc >/dev/null 2>&1; then
        nc -U "$sock_file"
    elif which socat >/dev/null 2>&1; then
        socat - "UNIX-CONNECT:$sock_file"
    else
        echo "Error: Neither 'nc' nor 'socat' installed to connect to unix socket."
        exit 1
    fi
}

case "$1" in
    start)
        start_cluster "$2" "$3" "$4"
        ;;
    stop)
        stop_cluster
        ;;
    status)
        status_cluster
        ;;
    console)
        connect_console "$2"
        ;;
    *)
        usage
        ;;
esac
