#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>

#define TCP_PORT 18515
#define IB_PORT 1
#define GID_INDEX 1
#define TOTAL_LLM_LAYERS 32

// --- DATA STRUCTURES ---

struct telemetry_data {
    uint32_t host_free_cpu_pct;
    uint32_t host_free_ram_mb;
    uint32_t host_assigned_layers;
    uint32_t vm_assigned_layers;
    uint32_t inference_queue_len;
};

struct rdma_connection_data {
    uint64_t addr;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t lid;
    union ibv_gid gid;
    uint32_t total_ram_mb; // The dynamic RAM capacity of the machine
};

// --- HARDWARE TELEMETRY ---

unsigned long long prev_user, prev_nice, prev_system, prev_idle, prev_iowait, prev_irq, prev_softirq;

void init_cpu_stats() {
    FILE *fp = fopen("/proc/stat", "r");
    if (fp) {
        fscanf(fp, "cpu %llu %llu %llu %llu %llu %llu %llu",
               &prev_user, &prev_nice, &prev_system, &prev_idle, &prev_iowait, &prev_irq, &prev_softirq);
        fclose(fp);
    }
}

uint32_t get_real_cpu_usage() {
    unsigned long long user, nice, system, idle, iowait, irq, softirq;
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return 0;
    
    fscanf(fp, "cpu %llu %llu %llu %llu %llu %llu %llu",
           &user, &nice, &system, &idle, &iowait, &irq, &softirq);
    fclose(fp);

    unsigned long long prev_total = prev_user + prev_nice + prev_system + prev_idle + prev_iowait + prev_irq + prev_softirq;
    unsigned long long current_total = user + nice + system + idle + iowait + irq + softirq;
    
    unsigned long long prev_active = prev_total - prev_idle;
    unsigned long long current_active = current_total - idle;

    unsigned long long total_diff = current_total - prev_total;
    unsigned long long active_diff = current_active - prev_active;

    prev_user = user; prev_nice = nice; prev_system = system; prev_idle = idle;
    prev_iowait = iowait; prev_irq = irq; prev_softirq = softirq;

    if (total_diff == 0) return 0;
    return (uint32_t)((active_diff * 100.0) / total_diff);
}

uint32_t get_real_free_ram_mb() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char buffer[256];
    unsigned long long mem_available = 0;
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (sscanf(buffer, "MemAvailable: %llu kB", &mem_available) == 1) break;
    }
    fclose(fp);
    return (uint32_t)(mem_available / 1024);
}

uint32_t get_total_ram_mb() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char buffer[256];
    unsigned long long mem_total = 0;
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (sscanf(buffer, "MemTotal: %llu kB", &mem_total) == 1) break;
    }
    fclose(fp);
    return (uint32_t)(mem_total / 1024);
}

// --- ALLOCATION ALGORITHM ---

void calculate_dynamic_split(uint32_t host_ram_mb, uint32_t vm_ram_mb, struct telemetry_data *telemetry) {
    uint32_t total_ram = host_ram_mb + vm_ram_mb;
    if (total_ram == 0) return; 

    float host_ratio = (float)host_ram_mb / total_ram;
    
    telemetry->host_assigned_layers = (uint32_t)(TOTAL_LLM_LAYERS * host_ratio);
    telemetry->vm_assigned_layers = TOTAL_LLM_LAYERS - telemetry->host_assigned_layers;

    // Safety net to ensure both machines get at least 1 layer for distributed testing
    if (telemetry->host_assigned_layers == TOTAL_LLM_LAYERS) {
        telemetry->host_assigned_layers--;
        telemetry->vm_assigned_layers = 1;
    }
    if (telemetry->vm_assigned_layers == TOTAL_LLM_LAYERS) {
        telemetry->vm_assigned_layers--;
        telemetry->host_assigned_layers = 1;
    }
}

// --- TCP KEY EXCHANGE ---

struct rdma_connection_data exchange_keys_via_tcp(int is_server, const char *server_ip, struct rdma_connection_data *local_data) {
    int sockfd, connfd;
    struct sockaddr_in servaddr;
    struct rdma_connection_data remote_data;
    
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(TCP_PORT);
    
    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (is_server) {
        servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
        bind(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));
        listen(sockfd, 1);
        printf("[TCP] Waiting for VM to connect on port %d...\n", TCP_PORT);
        connfd = accept(sockfd, NULL, NULL);
        
        read(connfd, &remote_data, sizeof(struct rdma_connection_data));
        write(connfd, local_data, sizeof(struct rdma_connection_data));
        close(connfd);
    } else {
        servaddr.sin_addr.s_addr = inet_addr(server_ip);
        printf("[TCP] Connecting to Host %s...\n", server_ip);
        while (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
            usleep(100000);
        }
        write(sockfd, local_data, sizeof(struct rdma_connection_data));
        read(sockfd, &remote_data, sizeof(struct rdma_connection_data));
    }
    close(sockfd);
    printf("[TCP] Handshake Complete! Remote QP: 0x%x | Remote RAM: %u MB\n", remote_data.qp_num, remote_data.total_ram_mb);
    return remote_data;
}

// --- QUEUE PAIR TRANSITIONS ---

int modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state        = IBV_QPS_INIT,
        .pkey_index      = 0,
        .port_num        = IB_PORT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC
    };
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

int modify_qp_to_rtr(struct ibv_qp *qp, struct rdma_connection_data *remote) {
    struct ibv_qp_attr attr = {
        .qp_state           = IBV_QPS_RTR,
        .path_mtu           = IBV_MTU_1024,
        .dest_qp_num        = remote->qp_num,
        .rq_psn             = 0,
        .max_dest_rd_atomic = 1,
        .min_rnr_timer      = 12,
        .ah_attr            = { .is_global = 1, .dlid = remote->lid, .sl = 0, .src_path_bits = 0, .port_num = IB_PORT }
    };
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    attr.ah_attr.grh.hop_limit = 1;
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state      = IBV_QPS_RTS,
        .timeout       = 14,
        .retry_cnt     = 7,
        .rnr_retry     = 7,
        .sq_psn        = 0,
        .max_rd_atomic = 1
    };
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

int push_telemetry_to_remote(struct ibv_qp *qp, struct ibv_mr *local_mr, struct telemetry_data *local_data, uint64_t remote_addr, uint32_t remote_rkey) {
    struct ibv_sge list = { .addr = (uintptr_t)local_data, .length = sizeof(struct telemetry_data), .lkey = local_mr->lkey };
    struct ibv_send_wr wr = {
        .wr_id = 1, .sg_list = &list, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE, .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = remote_addr, .wr.rdma.rkey = remote_rkey
    };
    struct ibv_send_wr *bad_wr;
    return ibv_post_send(qp, &wr, &bad_wr);
}

// --- MAIN EXECUTION ---

int main(int argc, char *argv[]) {
    int is_server = (argc == 1);
    const char *server_ip = is_server ? "192.168.100.1" : argv[1];

    struct ibv_device **dev_list;
    int num_devices;
    
    printf("[RDMA] 1. Initializing Device...\n");
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "CRITICAL ERROR: No RDMA devices found by libibverbs.\n");
        return 1;
    }

    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    if (!ctx) return 1;
    
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct telemetry_data *my_telemetry = calloc(1, sizeof(struct telemetry_data));
    struct ibv_mr *mr = ibv_reg_mr(pd, my_telemetry, sizeof(struct telemetry_data), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);

    struct ibv_cq *cq = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 }
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, IB_PORT, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(ctx, IB_PORT, GID_INDEX, &my_gid);

    // Pack our local details AND our total machine RAM to send across the wire
    struct rdma_connection_data my_data = { 
        .addr = (uintptr_t)mr->addr, 
        .rkey = mr->rkey, 
        .qp_num = qp->qp_num, 
        .lid = port_attr.lid, 
        .gid = my_gid,
        .total_ram_mb = get_total_ram_mb() 
    };
    
    struct rdma_connection_data remote_data = exchange_keys_via_tcp(is_server, server_ip, &my_data);

    printf("[RDMA] 2. Modifying Queue Pair States...\n");
    modify_qp_to_init(qp);
    modify_qp_to_rtr(qp, &remote_data);
    modify_qp_to_rts(qp);
    printf("[RDMA]    -> Link Established! (Ready to Send/Receive)\n\n");

    if (is_server) {
        printf("--- HOST MODE: Dynamic LLM Allocator Active ---\n");
        init_cpu_stats(); 
        
        // Dynamically set the VM's available RAM from the TCP handshake packet
        uint32_t vm_available_ram = remote_data.total_ram_mb; 
        printf("[INFO] Detected VM has %u MB of Total RAM.\n", vm_available_ram);

        while (1) {
            uint32_t live_cpu_usage = get_real_cpu_usage();
            uint32_t live_free_ram = get_real_free_ram_mb();

            my_telemetry->host_free_cpu_pct = 100 - live_cpu_usage;
            my_telemetry->host_free_ram_mb = live_free_ram;
            
            calculate_dynamic_split(live_free_ram, vm_available_ram, my_telemetry);
            
            // Write to RAM-disk for the local PyTorch script
            FILE *fp = fopen("/dev/shm/llm_split.json", "w");
            if (fp) {
                fprintf(fp, "{\"host_layers\": %u, \"vm_layers\": %u}\n", my_telemetry->host_assigned_layers, my_telemetry->vm_assigned_layers);
                fclose(fp);
            }

            // Blast the telemetry and allocation decision to the VM via RDMA
            push_telemetry_to_remote(qp, mr, my_telemetry, remote_data.addr, remote_data.rkey);
            
            printf("ALLOCATION -> Host RAM: %u MB | VM RAM: %u MB || SPLIT -> Host Layers: %u | VM Layers: %u\n", 
                   live_free_ram, vm_available_ram, my_telemetry->host_assigned_layers, my_telemetry->vm_assigned_layers);
            
            usleep(1000000); 
        }
    } else {
        printf("--- VM MODE: Awaiting Architecture Assignment ---\n");
        while (1) {
            printf("Master Node assigned me %u layers. (Host has %u layers). Free Host CPU: %d%%\n", 
                   my_telemetry->vm_assigned_layers, my_telemetry->host_assigned_layers, my_telemetry->host_free_cpu_pct);
            
            // Write to RAM-disk for the VM's local PyTorch script
            FILE *fp = fopen("/dev/shm/llm_split.json", "w");
            if (fp) {
                fprintf(fp, "{\"host_layers\": %u, \"vm_layers\": %u}\n", my_telemetry->host_assigned_layers, my_telemetry->vm_assigned_layers);
                fclose(fp);
            }
            usleep(1000000); 
        }
    }
    return 0;
}