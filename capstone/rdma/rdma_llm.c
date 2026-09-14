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

#define TCP_PORT 18516
#define IB_PORT 1
#define GID_INDEX 0
#define SHARED_MEM_SIZE (28 + (1024 * 3584 * 4)) // 14,680,092 bytes

// --- DATA STRUCTURES ---

struct llm_shared_pool {
    volatile int32_t job_status; // volatile forces CPU to check RAM, not cache
    volatile int32_t current_layer;
    volatile int32_t current_seq_len;
    volatile int32_t generated_token_id;
    volatile int32_t status;
    volatile int32_t host_layers;
    volatile int32_t vm_layers;
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
        
        if (read(connfd, &remote_data, sizeof(struct rdma_connection_data)) <= 0) {
            perror("read failed");
        }
        if (write(connfd, local_data, sizeof(struct rdma_connection_data)) <= 0) {
            perror("write failed");
        }
        close(connfd);
    } else {
        servaddr.sin_addr.s_addr = inet_addr(server_ip);
        printf("[TCP] Connecting to Host %s...\n", server_ip);
        while (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
            usleep(100000);
        }
        if (write(sockfd, local_data, sizeof(struct rdma_connection_data)) <= 0) {
            perror("write failed");
        }
        if (read(sockfd, &remote_data, sizeof(struct rdma_connection_data)) <= 0) {
            perror("read failed");
        }
    }
    close(sockfd);

    printf("[TCP] Handshake Complete!\n");
    return remote_data;
}

// --- QUEUE PAIR SETUP ---

void modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = { .qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = IB_PORT, .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

void modify_qp_to_rtr(struct ibv_qp *qp, struct rdma_connection_data *remote) {
    struct ibv_qp_attr attr = { .qp_state = IBV_QPS_RTR, .path_mtu = IBV_MTU_1024, .dest_qp_num = remote->qp_num, .rq_psn = 0, .max_dest_rd_atomic = 1, .min_rnr_timer = 12, .ah_attr = { .is_global = 1, .dlid = remote->lid, .sl = 0, .src_path_bits = 0, .port_num = IB_PORT } };
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    attr.ah_attr.grh.hop_limit = 1;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

void modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = { .qp_state = IBV_QPS_RTS, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = 0, .max_rd_atomic = 1 };
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

void push_memory_to_remote(struct ibv_qp *qp, struct ibv_mr *mr, struct llm_shared_pool *pool, uint64_t remote_addr, uint32_t remote_rkey) {
    struct ibv_sge list = { .addr = (uintptr_t)pool, .length = SHARED_MEM_SIZE, .lkey = mr->lkey };
    struct ibv_send_wr wr = { .wr_id = 1, .sg_list = &list, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE, .send_flags = IBV_SEND_SIGNALED, .wr.rdma.remote_addr = remote_addr, .wr.rdma.rkey = remote_rkey };
    struct ibv_send_wr *bad_wr;
    ibv_post_send(qp, &wr, &bad_wr);
}

// --- MAIN EXECUTION ---

int main(int argc, char *argv[]) {
    int is_host = (argc == 1);
    const char *server_ip = is_host ? "192.168.100.1" : argv[1];

    printf("[BRIDGE] 1. Creating Shared Memory Mailbox...\n");
    int fd = shm_open("llm_buffer", O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("CRITICAL ERROR: shm_open failed");
        return 1;
    }

    if (ftruncate(fd, SHARED_MEM_SIZE) == -1) {
        perror("CRITICAL ERROR: ftruncate failed");
        return 1;
    }
    
    struct llm_shared_pool *pool = mmap(NULL, SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pool == MAP_FAILED) {
        perror("CRITICAL ERROR: mmap failed");
        return 1;
    }
    
    memset(pool, 0, SHARED_MEM_SIZE); // Safe to clean slate now!

    printf("[BRIDGE] 2. Initializing RDMA Hardware...\n");
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "CRITICAL ERROR: No RDMA devices found. Is the driver loaded?\n");
        return 1;
    }

    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        fprintf(stderr, "CRITICAL ERROR: Could not open RDMA device.\n");
        return 1;
    }
    // struct ibv_device **dev_list = ibv_ge/t_device_list(NULL);
    // struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    
    // THE MAGIC: Registering the Shared Memory file directly to the Network Card!
    struct ibv_mr *mr = ibv_reg_mr(pd, pool, SHARED_MEM_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    struct ibv_cq *cq = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_attr = { .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC, .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 } };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, IB_PORT, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(ctx, IB_PORT, GID_INDEX, &my_gid);

    struct rdma_connection_data my_data = { .addr = (uintptr_t)pool, .rkey = mr->rkey, .qp_num = qp->qp_num, .lid = port_attr.lid, .gid = my_gid };
    struct rdma_connection_data remote = exchange_keys_via_tcp(is_host, server_ip, &my_data);

    modify_qp_to_init(qp);
    modify_qp_to_rtr(qp, &remote);
    modify_qp_to_rts(qp);
    
    printf("[BRIDGE] Link Established. RDMA Bridge is ARMED.\n");

    // ==========================================
    // THE SYNCHRONIZATION LOOP
    // ==========================================
    struct ibv_wc wc;
    
    if (is_host) {
        printf("[RDMA-HOST] Waiting for PyTorch (Layers 0-23)...\n");
        while (1) {
            if (pool->job_status == 1) { 
                printf("[RDMA-HOST] Blasting 14MB tensor to VM...\n");
                push_memory_to_remote(qp, mr, pool, remote.addr, remote.rkey);
                
                // Wait for the card to confirm it sent the data
                while (ibv_poll_cq(cq, 1, &wc) < 1);
                
                // Wait for the VM RDMA card to overwrite our RAM with the answer
                // Wait for the VM RDMA card to overwrite our RAM with the answer
                while (pool->job_status != 2) usleep(500); 
                
                // 1. Print the token using your perfectly formatted struct
                printf("[RDMA-HOST] Token %d received from VM.\n", pool->generated_token_id);

                // 2. Wake up the Host Python script
                // Note: Change '3' to whatever number your Python script is waiting for!
                // (It might also be waiting for pool->status = 1 instead)
                pool->job_status = 2;// Change this to whatever status integer Python is waiting for!
            }
            usleep(500);
        }
    } else {
        printf("[RDMA-VM] Waiting for Host RDMA payload...\n");
        while (1) {
            if (pool->job_status == 1) { 
                printf("[RDMA-VM] Tensor arrived! Waking up VM PyTorch...\n");
                
                // Wait for VM Python to process layers 24-27 and change status to 2
                while (pool->job_status != 2) usleep(500); 
                
                printf("[RDMA-VM] Sending Token %u back to Host...\n", pool->generated_token_id);
                push_memory_to_remote(qp, mr, pool, remote.addr, remote.rkey);
                
                // Wait for the card to confirm send
                while (ibv_poll_cq(cq, 1, &wc) < 1);
                
                pool->job_status = 0;
            }
            usleep(500);
        }
    }
    return 0;
}