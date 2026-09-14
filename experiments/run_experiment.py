import os
import sys
import time
import json
import argparse
import subprocess
try:
    import numpy as np
except ImportError:
    import statistics
    class np:
        @staticmethod
        def mean(arr): return float(statistics.mean(arr)) if arr else 0.0
        @staticmethod
        def std(arr): return float(statistics.stdev(arr)) if len(arr) > 1 else 0.0



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, "capstone", "rdma", "distributed_pipeline.py")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_local_pipeline(total_nodes, backend, seq_len, max_tokens, port_base=19000):
    """
    Runs an N-node pipeline locally using separate processes.
    Enables benchmarking the exact communication backends and measuring delay breakdowns.
    """
    node_ips = ["127.0.0.1"] * total_nodes
    node_ips_str = ",".join(node_ips)
    json_output = os.path.join(RESULTS_DIR, f"temp_run_{backend}_n{total_nodes}.json")

    processes = []
    print(f"\n---> Spawning {total_nodes} pipeline nodes locally (Backend: {backend.upper()})...")

    try:
        # Spawn worker nodes first (Rank 1 to total_nodes - 1)
        for rank in range(1, total_nodes):
            cmd = [
                sys.executable, PIPELINE_SCRIPT,
                "--rank", str(rank),
                "--total-nodes", str(total_nodes),
                "--node-ips", node_ips_str,
                "--backend", backend,
                "--port-base", str(port_base),
                "--benchmark",
                "--seq-len", str(seq_len),
                "--max-tokens", str(max_tokens)
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(p)

        # Give workers a moment to bind sockets
        time.sleep(1.0)

        # Spawn Master node (Rank 0)
        cmd_master = [
            sys.executable, PIPELINE_SCRIPT,
            "--rank", "0",
            "--total-nodes", str(total_nodes),
            "--node-ips", node_ips_str,
            "--backend", backend,
            "--port-base", str(port_base),
            "--benchmark",
            "--seq-len", str(seq_len),
            "--max-tokens", str(max_tokens),
            "--json-output", json_output
        ]
        p_master = subprocess.Popen(cmd_master, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p_master)

        # Wait for master to finish
        stdout, stderr = p_master.communicate(timeout=180)
        
        # Wait for remaining processes
        for p in processes[:-1]:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

        if os.path.exists(json_output):
            with open(json_output, "r") as f:
                data = json.load(f)
            try: os.remove(json_output)
            except: pass
            return data
        else:
            print(f"Warning: Master did not produce JSON output. Stderr: {stderr.decode()[:400]}")
            return None

    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()

def run_cluster_pipeline(total_nodes, backend, seq_len, max_tokens, cluster_mgr):
    """
    Drives an actual running QEMU cluster via cluster_mgr.sh and SSH/direct socket.
    """
    print(f"\n---> Running on QEMU Cluster with {total_nodes} nodes (Backend: {backend.upper()})...")
    # Generate node IPs: 192.168.100.1 (Host/Rank 0) or 192.168.100.2..N+1 (VMs)
    node_ips = ["192.168.100.1"] + [f"192.168.100.{i+1}" for i in range(1, total_nodes)]
    node_ips_str = ",".join(node_ips)
    json_output = os.path.join(RESULTS_DIR, f"cluster_run_{backend}_n{total_nodes}.json")

    # Command for Host Master (Rank 0)
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "--rank", "0",
        "--total-nodes", str(total_nodes),
        "--node-ips", node_ips_str,
        "--backend", backend,
        "--benchmark",
        "--seq-len", str(seq_len),
        "--max-tokens", str(max_tokens),
        "--json-output", json_output
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate(timeout=180)

    if os.path.exists(json_output):
        with open(json_output, "r") as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Latency & Protocol Benchmark Harness")
    parser.add_argument("--nodes", type=str, default="2,3,4,5", help="Comma-separated node counts to test (e.g. 2,3,4,5)")
    parser.add_argument("--backends", type=str, default="tcp,rdma", help="Backends to benchmark (tcp, rdma, or tcp,rdma)")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for token generation")
    parser.add_argument("--max-tokens", type=int, default=15, help="Number of tokens to generate per trial")
    parser.add_argument("--trials", type=int, default=3, help="Number of repetitions per configuration")
    parser.add_argument("--mode", choices=["local", "cluster"], default="local",
                        help="Execution mode: 'local' (host multi-process benchmark) or 'cluster' (QEMU VMs)")

    args = parser.parse_args()
    node_counts = [int(n.strip()) for n in args.nodes.split(",") if n.strip()]
    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]

    print("==================================================================")
    print("DISTRIBUTED LLM INFERENCE BENCHMARK HARNESS")
    print(f"Node Counts Tested : {node_counts}")
    print(f"Backends Tested    : {backends}")
    print(f"Tokens Per Run     : {args.max_tokens}")
    print(f"Sequence Length    : {args.seq_len} tokens (~{args.seq_len * 3584 * 4 / (1024*1024):.2f} MB payload)")
    print(f"Trials Per Config  : {args.trials}")
    print(f"Execution Mode     : {args.mode.upper()}")
    print("==================================================================")

    all_results = []
    port_counter = 19000

    for num_nodes in node_counts:
        for backend in backends:
            print(f"\n==================================================")
            print(f"BENCHMARKING: {num_nodes} NODES | BACKEND: {backend.upper()}")
            print(f"==================================================")

            trial_latencies = []
            trial_computes = []
            trial_networks = []
            trial_throughputs = []

            for trial in range(args.trials):
                port_counter += (num_nodes + 5)
                print(f"[Trial {trial + 1}/{args.trials}] Running test...", end="", flush=True)

                if args.mode == "local":
                    # In local benchmark mode, for 'rdma', if running without physical RDMA hardware,
                    # we profile the shm/zero-copy communication path
                    res = run_local_pipeline(num_nodes, backend, args.seq_len, args.max_tokens, port_base=port_counter)
                else:
                    res = run_cluster_pipeline(num_nodes, backend, args.seq_len, args.max_tokens, None)

                if res:
                    trial_latencies.append(res['mean_step_latency_ms'])
                    trial_computes.append(res['mean_compute_latency_ms'])
                    trial_networks.append(res['mean_network_latency_ms'])
                    trial_throughputs.append(res['tokens_per_second'])
                    print(f" Done! Mean Latency: {res['mean_step_latency_ms']:.2f} ms (Compute: {res['mean_compute_latency_ms']:.2f} ms | Network: {res['mean_network_latency_ms']:.2f} ms)")
                else:
                    print(" FAILED")

            if trial_latencies:
                mean_lat = float(np.mean(trial_latencies))
                std_lat = float(np.std(trial_latencies))
                mean_comp = float(np.mean(trial_computes))
                mean_net = float(np.mean(trial_networks))
                mean_tps = float(np.mean(trial_throughputs))
                per_hop_net = mean_net / max(1, (num_nodes - 1))

                summary_entry = {
                    "num_nodes": num_nodes,
                    "backend": backend,
                    "seq_len": args.seq_len,
                    "payload_size_mb": round(args.seq_len * 3584 * 4 / (1024 * 1024), 2),
                    "mean_latency_ms": round(mean_lat, 2),
                    "std_latency_ms": round(std_lat, 2),
                    "compute_time_ms": round(mean_comp, 2),
                    "network_time_ms": round(mean_net, 2),
                    "per_hop_delay_ms": round(per_hop_net, 2),
                    "throughput_tps": round(mean_tps, 2)
                }
                all_results.append(summary_entry)

    # Save overall results to JSON
    output_json = os.path.join(RESULTS_DIR, "benchmark_data.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV summary
    output_csv = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    with open(output_csv, "w") as f:
        f.write("Nodes,Backend,Payload_MB,Mean_Latency_ms,Std_Latency_ms,Compute_ms,Network_ms,Per_Hop_Delay_ms,Throughput_tok_s\n")
        for r in all_results:
            f.write(f"{r['num_nodes']},{r['backend']},{r['payload_size_mb']},{r['mean_latency_ms']},{r['std_latency_ms']},{r['compute_time_ms']},{r['network_time_ms']},{r['per_hop_delay_ms']},{r['throughput_tps']}\n")

    print("\n==================================================================")
    print("BENCHMARK COMPLETE!")
    print(f"Results saved to:")
    print(f"  - JSON: {output_json}")
    print(f"  - CSV : {output_csv}")
    print("==================================================================\n")

    # Run plot script to display comparison table and generate graphs
    plot_script = os.path.join(SCRIPT_DIR, "plot_results.py")
    if os.path.exists(plot_script):
        subprocess.run([sys.executable, plot_script])

if __name__ == "__main__":
    main()
