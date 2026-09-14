#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>

#define BUFFER_SIZE 1024
#define TCP_PORT 18515
#define IB_PORT 1
#define GID_INDEX 1 // IPv4 GID we verified earlier

// --- DATA STRUCTURES ---

struct telemetry_data {
    uint32_t host_free_cpu_pct;
    uint32_t host_free_ram_mb;
    uint32_t inference_queue_len;
    uint32_t lock_flag;
};

struct rdma_connection_data {
    uint64_t addr;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t lid;
    union ibv_gid gid;
};

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
    printf("[TCP] Handshake Complete! Remote QP: 0x%x\n", remote_data.qp_num);
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
    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    return ibv_modify_qp(qp, &attr, flags);
}

int modify_qp_to_rtr(struct ibv_qp *qp, struct rdma_connection_data *remote) {
    struct ibv_qp_attr attr = {
        .qp_state           = IBV_QPS_RTR,
        .path_mtu           = IBV_MTU_1024,
        .dest_qp_num        = remote->qp_num,
        .rq_psn             = 0,
        .max_dest_rd_atomic = 1,
        .min_rnr_timer      = 12,
        .ah_attr            = {
            .is_global      = 1,
            .dlid           = remote->lid,
            .sl             = 0,
            .src_path_bits  = 0,
            .port_num       = IB_PORT
        }
    };
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    attr.ah_attr.grh.hop_limit = 1;

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    return ibv_modify_qp(qp, &attr, flags);
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
    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    return ibv_modify_qp(qp, &attr, flags);
}

// --- TELEMETRY LOGIC ---

int push_telemetry_to_remote(struct ibv_qp *qp, struct ibv_mr *local_mr, struct telemetry_data *local_data, uint64_t remote_addr, uint32_t remote_rkey) {
    struct ibv_sge list = {
        .addr   = (uintptr_t)local_data,
        .length = sizeof(struct telemetry_data),
        .lkey   = local_mr->lkey
    };

    struct ibv_send_wr wr = {
        .wr_id      = 1,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = remote_addr,
        .wr.rdma.rkey        = remote_rkey
    };

    struct ibv_send_wr *bad_wr;
    return ibv_post_send(qp, &wr, &bad_wr);
}

// --- MAIN EXECUTION ---

int main(int argc, char *argv[]) {
    int is_server = (argc == 1);
    const char *server_ip = "192.168.100.1";
    if (!is_server) server_ip = argv[1];

    struct ibv_device **dev_list;
    int num_devices;
    
    printf("[RDMA] 1. Initializing Device...\n");
    dev_list = ibv_get_device_list(&num_devices);
    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    // Register 1KB of RAM for our telemetry struct
    struct telemetry_data *my_telemetry = calloc(1, sizeof(struct telemetry_data));
    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    struct ibv_mr *mr = ibv_reg_mr(pd, my_telemetry, sizeof(struct telemetry_data), access_flags);

    struct ibv_cq *cq = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 }
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    // Get local routing info
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, IB_PORT, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(ctx, IB_PORT, GID_INDEX, &my_gid);

    // Prepare business card
    struct rdma_connection_data my_data = {
        .addr = (uintptr_t)mr->addr,
        .rkey = mr->rkey,
        .qp_num = qp->qp_num,
        .lid = port_attr.lid,
        .gid = my_gid
    };

    // Swap Keys
    struct rdma_connection_data remote_data = exchange_keys_via_tcp(is_server, server_ip, &my_data);

    // Bring up the RDMA Link
    printf("[RDMA] 2. Modifying Queue Pair States...\n");
    modify_qp_to_init(qp);
    modify_qp_to_rtr(qp, &remote_data);
    modify_qp_to_rts(qp);
    printf("[RDMA]    -> Link Established! (Ready to Send/Receive)\n\n");

    // The Action Loop
    if (is_server) {
        printf("--- HOST MODE: Pushing Telemetry to VM ---\n");
        uint32_t simulated_cpu = 100;
        while (1) {
            my_telemetry->host_free_cpu_pct = simulated_cpu;
            my_telemetry->host_free_ram_mb = 16384;
            my_telemetry->inference_queue_len = 0;
            
            // SILENTLY write this data into the VM's RAM
            push_telemetry_to_remote(qp, mr, my_telemetry, remote_data.addr, remote_data.rkey);
            
            printf("Pushed CPU: %d%%\n", simulated_cpu);
            simulated_cpu--;
            if(simulated_cpu < 50) simulated_cpu = 100;
            
            usleep(500000); // Wait 0.5s
        }
    } else {
        printf("--- VM MODE: Watching Local Memory ---\n");
        while (1) {
            // We just read our local RAM. The Host is updating it over the network!
            printf("Host CPU is currently: %d%% Free\n", my_telemetry->host_free_cpu_pct);
            usleep(500000); // Wait 0.5s
        }
    }

    return 0;
}