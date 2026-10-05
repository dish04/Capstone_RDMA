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
#define HEADER_SIZE 28
#define HIDDEN_DIM 3584
#define MAX_SEQ_LEN 4096
#define SHARED_MEM_SIZE (HEADER_SIZE + (MAX_SEQ_LEN * HIDDEN_DIM * 4)) // matches distributed_pipeline.py

#define FLAG_IDLE 0
#define FLAG_TENSOR_READY 1
#define FLAG_TOKEN_READY 2
#define FLAG_TERMINATE 99

// Header & Shared Memory Mailbox structure (28 bytes header + tensor data)
struct llm_shared_pool {
    volatile int32_t job_status;      // 0 = Idle, 1 = Tensor Ready, 2 = Token Ready, 99 = Terminate
    volatile int32_t current_layer;
    volatile int32_t current_seq_len;
    volatile int32_t generated_token_id;
    volatile int32_t status;
    volatile int32_t total_nodes;
    volatile int32_t rank;            // sender rank
    // Tensor data follows immediately
};

struct rdma_connection_data {
    uint64_t addr;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t lid;
    union ibv_gid gid;
};

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

// --- SHARED MEMORY ---
// Rank r uses /dev/shm/<name>_in_<r> (written remotely by the predecessor)
// and /dev/shm/<name>_out_<r> (filled by local Python, pushed to the successor).
static struct llm_shared_pool *map_shm(const char *name) {
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) die("shm_open failed");
    if (ftruncate(fd, SHARED_MEM_SIZE) == -1) die("ftruncate failed");
    struct llm_shared_pool *pool = mmap(NULL, SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pool == MAP_FAILED) die("mmap failed");
    close(fd);
    return pool;
}

// --- TCP KEY EXCHANGE ---
static void read_full(int fd, void *buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, (char *)buf + done, len - done);
        if (n <= 0) die("read failed");
        done += n;
    }
}

static void write_full(int fd, const void *buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, (const char *)buf + done, len - done);
        if (n <= 0) die("write failed");
        done += n;
    }
}

static int listen_on(int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) die("socket failed");
    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) die("bind failed");
    if (listen(sockfd, 1) < 0) die("listen failed");
    printf("[TCP] Listening for predecessor on port %d...\n", port);
    return sockfd;
}

static int connect_to(const char *ip, int port) {
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = inet_addr(ip);

    printf("[TCP] Connecting to successor %s:%d...\n", ip, port);
    while (1) {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) die("socket failed");
        if (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == 0) return sockfd;
        close(sockfd); // a failed connect leaves the socket unusable, so retry with a fresh one
        usleep(100000); // 100ms retry
    }
}

// --- QUEUE PAIR TRANSITIONS ---
static void modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_INIT,
        .pkey_index = 0,
        .port_num = IB_PORT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
    };
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        die("ibv_modify_qp INIT failed");
}

static void modify_qp_to_rtr(struct ibv_qp *qp, struct rdma_connection_data *remote) {
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
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        die("ibv_modify_qp RTR failed");
}

static void modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {
        .qp_state = IBV_QPS_RTS,
        .timeout = 14,
        .retry_cnt = 7,
        .rnr_retry = 7,
        .sq_psn = 0,
        .max_rd_atomic = 1
    };
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
        die("ibv_modify_qp RTS failed");
}

static void connect_qp(struct ibv_qp *qp, struct rdma_connection_data *remote) {
    modify_qp_to_init(qp);
    modify_qp_to_rtr(qp, remote);
    modify_qp_to_rts(qp);
}

static struct ibv_qp *create_qp(struct ibv_pd *pd, struct ibv_cq *cq) {
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 }
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);
    if (!qp) die("ibv_create_qp failed");
    return qp;
}

// Pushes the outbox into the successor's inbox: tensor first, header last.
// Both writes go on the same RC QP, so they land in order and the receiver
// never sees the header flag before the tensor bytes.
static void push_message(struct ibv_qp *qp, struct ibv_cq *cq, struct ibv_mr *out_mr,
                         struct rdma_connection_data *remote, size_t tensor_bytes) {
    char *base = (char *)out_mr->addr;
    struct ibv_sge tensor_sge = { .addr = (uintptr_t)(base + HEADER_SIZE), .length = tensor_bytes, .lkey = out_mr->lkey };
    struct ibv_sge header_sge = { .addr = (uintptr_t)base, .length = HEADER_SIZE, .lkey = out_mr->lkey };

    struct ibv_send_wr header_wr = {
        .wr_id = 2, .sg_list = &header_sge, .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE, .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = remote->addr, .wr.rdma.rkey = remote->rkey
    };
    struct ibv_send_wr tensor_wr = {
        .wr_id = 1, .next = &header_wr, .sg_list = &tensor_sge, .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .wr.rdma.remote_addr = remote->addr + HEADER_SIZE, .wr.rdma.rkey = remote->rkey
    };

    struct ibv_send_wr *bad_wr;
    if (ibv_post_send(qp, tensor_bytes > 0 ? &tensor_wr : &header_wr, &bad_wr))
        die("ibv_post_send failed");

    struct ibv_wc wc;
    int n;
    while ((n = ibv_poll_cq(cq, 1, &wc)) == 0);
    if (n < 0 || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "RDMA write failed: %s\n", n < 0 ? "poll error" : ibv_wc_status_str(wc.status));
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    int rank = 0;
    int total_nodes = 2;
    int port_base = 18516;
    const char *downstream_ip = "192.168.100.2";
    const char *shm_name = "llm_buffer";

    // Parse simple args. --downstream-ip is the successor's IP (the last rank points back to rank 0).
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-nodes") == 0 && i + 1 < argc) total_nodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--downstream-ip") == 0 && i + 1 < argc) downstream_ip = argv[++i];
        else if (strcmp(argv[i], "--port-base") == 0 && i + 1 < argc) port_base = atoi(argv[++i]);
        else if (strcmp(argv[i], "--shm-name") == 0 && i + 1 < argc) shm_name = argv[++i];
    }

    int prev_rank = (rank - 1 + total_nodes) % total_nodes;
    int next_rank = (rank + 1) % total_nodes;

    printf("====================================================\n");
    printf("[RDMA-PIPELINE] Rank %d of %d | Prev: Rank %d | Next: Rank %d @ %s\n",
           rank, total_nodes, prev_rank, next_rank, downstream_ip);
    printf("====================================================\n");

    // 1. Shared Memory: inbox (remote target) and outbox (local source)
    char inbox_name[256], outbox_name[256];
    snprintf(inbox_name, sizeof(inbox_name), "%s_in_%d", shm_name, rank);
    snprintf(outbox_name, sizeof(outbox_name), "%s_out_%d", shm_name, rank);
    struct llm_shared_pool *inbox = map_shm(inbox_name);
    struct llm_shared_pool *outbox = map_shm(outbox_name);

    // 2. IBVerbs Initialization
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "CRITICAL ERROR: No RDMA devices found by libibverbs.\n");
        return 1;
    }
    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    if (!ctx) die("ibv_open_device failed");
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) die("ibv_alloc_pd failed");

    struct ibv_mr *in_mr = ibv_reg_mr(pd, inbox, SHARED_MEM_SIZE,
                                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    struct ibv_mr *out_mr = ibv_reg_mr(pd, outbox, SHARED_MEM_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!in_mr || !out_mr) die("ibv_reg_mr failed");

    // Two connections per rank: one from the predecessor, one to the successor
    struct ibv_cq *cq_prev = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    struct ibv_cq *cq_next = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    if (!cq_prev || !cq_next) die("ibv_create_cq failed");
    struct ibv_qp *qp_prev = create_qp(pd, cq_prev);
    struct ibv_qp *qp_next = create_qp(pd, cq_next);

    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx, IB_PORT, &port_attr)) die("ibv_query_port failed");
    union ibv_gid my_gid;
    if (ibv_query_gid(ctx, IB_PORT, GID_INDEX, &my_gid)) die("ibv_query_gid failed");

    // What the predecessor needs: where to write (our inbox) and our QP for that link
    struct rdma_connection_data to_prev = {
        .addr = (uintptr_t)inbox, .rkey = in_mr->rkey,
        .qp_num = qp_prev->qp_num, .lid = port_attr.lid, .gid = my_gid
    };
    // What the successor needs: only our QP for that link (it never writes to us)
    struct rdma_connection_data to_next = {
        .addr = 0, .rkey = 0,
        .qp_num = qp_next->qp_num, .lid = port_attr.lid, .gid = my_gid
    };
    struct rdma_connection_data from_prev, from_next;

    // 3. Handshake around the ring. Order avoids deadlock: every rank listens,
    // connects to its successor and sends first, then serves its predecessor,
    // and only then waits for the successor's reply.
    int listen_fd = listen_on(port_base + rank);
    int next_fd = connect_to(downstream_ip, port_base + next_rank);
    write_full(next_fd, &to_next, sizeof(to_next));

    int prev_fd = accept(listen_fd, NULL, NULL);
    if (prev_fd < 0) die("accept failed");
    read_full(prev_fd, &from_prev, sizeof(from_prev));
    write_full(prev_fd, &to_prev, sizeof(to_prev));
    close(prev_fd);
    close(listen_fd);

    read_full(next_fd, &from_next, sizeof(from_next));
    close(next_fd);
    printf("[RDMA-PIPELINE] Handshakes complete (prev QP 0x%x, next QP 0x%x)\n",
           from_prev.qp_num, from_next.qp_num);

    connect_qp(qp_prev, &from_prev);
    connect_qp(qp_next, &from_next);
    printf("[RDMA-PIPELINE] Both Queue Pairs Armed and Ready!\n");

    // 4. Forwarding Loop: push each outbox message into the successor's inbox.
    // Only one message is in flight around the ring (rank 0 waits for the token),
    // so the successor's inbox is free whenever our outbox fills.
    printf("[RDMA-PIPELINE] Entering forwarding loop...\n");
    while (1) {
        int32_t flag = outbox->job_status;
        if (flag == FLAG_IDLE) {
            usleep(250);
            continue;
        }
        __sync_synchronize(); // read the header fields after seeing the flag

        size_t tensor_bytes = 0;
        if (flag == FLAG_TENSOR_READY) {
            tensor_bytes = (size_t)outbox->current_seq_len * HIDDEN_DIM * 4;
            if (tensor_bytes > SHARED_MEM_SIZE - HEADER_SIZE) tensor_bytes = SHARED_MEM_SIZE - HEADER_SIZE;
        }
        push_message(qp_next, cq_next, out_mr, &from_next, tensor_bytes);

        __sync_synchronize();
        outbox->job_status = FLAG_IDLE; // tell local Python the outbox is free
        if (flag == FLAG_TERMINATE) break;
    }

    return 0;
}
