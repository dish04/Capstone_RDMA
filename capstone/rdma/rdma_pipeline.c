#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>

#define IB_PORT 1
#define GID_INDEX 0
#define SHARED_MEM_SIZE (28 + (1024 * 3584 * 4)) // 14,680,092 bytes

// Header & Shared Memory Mailbox structure (28 bytes header + tensor data)
struct llm_shared_pool {
    volatile int32_t job_status;      // 0 = Idle, 1 = Forward Tensor Ready, 2 = Token Ready
    volatile int32_t current_layer;
    volatile int32_t current_seq_len;
    volatile int32_t generated_token_id;
    volatile int32_t status;
    volatile int32_t total_nodes;
    volatile int32_t rank;
    // Tensor data follows immediately
};

struct rdma_connection_data {
    uint64_t addr;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t lid;
    union ibv_gid gid;
};

// --- TCP KEY EXCHANGE ---
struct rdma_connection_data exchange_keys_server(int port, struct rdma_connection_data *local_data) {
    int sockfd, connfd;
    struct sockaddr_in servaddr;
    struct rdma_connection_data remote_data;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);

    bind(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));
    listen(sockfd, 1);
    printf("[TCP-Server] Awaiting RDMA handshake on port %d...\n", port);
    connfd = accept(sockfd, NULL, NULL);

    read(connfd, &remote_data, sizeof(struct rdma_connection_data));
    write(connfd, local_data, sizeof(struct rdma_connection_data));
    close(connfd);
    close(sockfd);
    printf("[TCP-Server] Handshake on port %d successful!\n", port);
    return remote_data;
}

struct rdma_connection_data exchange_keys_client(const char *server_ip, int port, struct rdma_connection_data *local_data) {
    int sockfd;
    struct sockaddr_in servaddr;
    struct rdma_connection_data remote_data;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = inet_addr(server_ip);

    printf("[TCP-Client] Connecting to %s:%d for RDMA handshake...\n", server_ip, port);
    while (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        usleep(100000); // 100ms retry
    }

    write(sockfd, local_data, sizeof(struct rdma_connection_data));
    read(sockfd, &remote_data, sizeof(struct rdma_connection_data));
    close(sockfd);
    printf("[TCP-Client] Handshake to %s:%d successful!\n", server_ip, port);
    return remote_data;
}

// --- QUEUE PAIR TRANSITIONS ---
void modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_INIT,
        .pkey_index = 0,
        .port_num = IB_PORT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
    };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

void modify_qp_to_rtr(struct ibv_qp *qp, struct rdma_connection_data *remote) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = IBV_MTU_1024,
        .dest_qp_num = remote->qp_num,
        .rq_psn = 0,
        .max_dest_rd_atomic = 1,
        .min_rnr_timer = 12,
        .ah_attr = {
            .is_global = 1,
            .dlid = remote->lid,
            .sl = 0,
            .src_path_bits = 0,
            .port_num = IB_PORT,
            .grh = {
                .dgid = remote->gid,
                .sgid_index = GID_INDEX,
                .hop_limit = 1
            }
        }
    };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

void modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_RTS,
        .timeout = 14,
        .retry_cnt = 7,
        .rnr_retry = 7,
        .sq_psn = 0,
        .max_rd_atomic = 1
    };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

void push_memory_to_remote(struct ibv_qp *qp, struct ibv_mr *mr, struct llm_shared_pool *pool, uint64_t remote_addr, uint32_t remote_rkey, size_t bytes) {
    struct ibv_sge list = { .addr = (uintptr_t)pool, .length = bytes, .lkey = mr->lkey };
    struct ibv_send_wr wr = {
        .wr_id = 1,
        .sg_list = &list,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = remote_addr,
        .wr.rdma.rkey = remote_rkey
    };
    struct ibv_send_wr *bad_wr;
    ibv_post_send(qp, &wr, &bad_wr);
}

int main(int argc, char *argv[]) {
    int rank = 0;
    int total_nodes = 2;
    int port_base = 18516;
    const char *downstream_ip = "192.168.100.2";

    // Parse simple args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-nodes") == 0 && i + 1 < argc) total_nodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--downstream-ip") == 0 && i + 1 < argc) downstream_ip = argv[++i];
        else if (strcmp(argv[i], "--port-base") == 0 && i + 1 < argc) port_base = atoi(argv[++i]);
    }

    printf("====================================================\n");
    printf("[RDMA-PIPELINE] Rank %d of %d | Downstream IP: %s\n", rank, total_nodes, downstream_ip);
    printf("====================================================\n");

    // 1. Shared Memory Mailbox Setup
    int fd = shm_open("llm_buffer", O_CREAT | O_RDWR, 0666);
    if (fd == -1) { perror("shm_open failed"); return 1; }
    ftruncate(fd, SHARED_MEM_SIZE);

    struct llm_shared_pool *pool = mmap(NULL, SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pool == MAP_FAILED) { perror("mmap failed"); return 1; }
    memset(pool, 0, SHARED_MEM_SIZE);

    // 2. IBVerbs Initialization
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "CRITICAL ERROR: No RDMA devices found by libibverbs.\n");
        return 1;
    }
    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_mr *mr = ibv_reg_mr(pd, pool, SHARED_MEM_SIZE,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    struct ibv_cq *cq_down = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq_down, .recv_cq = cq_down, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 }
    };
    struct ibv_qp *qp_down = ibv_create_qp(pd, &qp_attr);

    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, IB_PORT, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(ctx, IB_PORT, GID_INDEX, &my_gid);

    struct rdma_connection_data my_data = {
        .addr = (uintptr_t)pool,
        .rkey = mr->rkey,
        .qp_num = qp_down->qp_num,
        .lid = port_attr.lid,
        .gid = my_gid
    };

    // 3. Setup Downstream Connection
    // Even ranks act as servers, odd act as clients or sequential handshake
    int downstream_port = port_base + rank;
    struct rdma_connection_data remote_down;

    if (rank == 0) {
        // Master: acts as server for downstream connection
        remote_down = exchange_keys_server(downstream_port, &my_data);
    } else {
        // Workers: connect to upstream or downstream
        remote_down = exchange_keys_client(downstream_ip, port_base + (rank - 1), &my_data);
    }

    modify_qp_to_init(qp_down);
    modify_qp_to_rtr(qp_down, &remote_down);
    modify_qp_to_rts(qp_down);
    printf("[RDMA-PIPELINE] Queue Pair Armed and Ready!\n");

    // 4. Forwarding Loop
    struct ibv_wc wc;
    printf("[RDMA-PIPELINE] Entering synchronization loop...\n");

    while (1) {
        if (pool->job_status == 1) {
            size_t send_size = 28 + (pool->current_seq_len * 3584 * 4);
            if (send_size > SHARED_MEM_SIZE) send_size = SHARED_MEM_SIZE;

            // Push forward to downstream node
            push_memory_to_remote(qp_down, mr, pool, remote_down.addr, remote_down.rkey, send_size);
            while (ibv_poll_cq(cq_down, 1, &wc) < 1);
            pool->job_status = 0;
        } else if (pool->job_status == 2 && rank > 0) {
            // Push token back to Rank 0
            push_memory_to_remote(qp_down, mr, pool, remote_down.addr, remote_down.rkey, 28);
            while (ibv_poll_cq(cq_down, 1, &wc) < 1);
            pool->job_status = 0;
        }
        usleep(250);
    }

    return 0;
}
