#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>

#define TCP_PORT 18516
#define IB_PORT 1
#define GID_INDEX 1
#define VECTOR_SIZE 5 // Reduced size for clear terminal printing

struct shared_memory_pool {
    volatile uint32_t job_status; 
    float input_vector[VECTOR_SIZE];
    float output_vector[VECTOR_SIZE];
};

struct rdma_connection_data {
    uint64_t addr;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t lid;
    union ibv_gid gid;
};

// --- TCP KEY EXCHANGE (OOB) ---
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
        printf("[TCP] Waiting for VM on port %d...\n", TCP_PORT);
        connfd = accept(sockfd, NULL, NULL);
        read(connfd, &remote_data, sizeof(struct rdma_connection_data));
        write(connfd, local_data, sizeof(struct rdma_connection_data));
        close(connfd);
    } else {
        servaddr.sin_addr.s_addr = inet_addr(server_ip);
        printf("[TCP] Connecting to Host %s...\n", server_ip);
        while (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) usleep(100000);
        write(sockfd, local_data, sizeof(struct rdma_connection_data));
        read(sockfd, &remote_data, sizeof(struct rdma_connection_data));
    }
    close(sockfd);
    return remote_data;
}

// --- QP TRANSITIONS (Standard RC) ---
int modify_qp_to_init(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = { .qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = IB_PORT, .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE };
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

int modify_qp_to_rtr(struct ibv_qp *qp, struct rdma_connection_data *remote) {
    struct ibv_qp_attr attr = { .qp_state = IBV_QPS_RTR, .path_mtu = IBV_MTU_1024, .dest_qp_num = remote->qp_num, .rq_psn = 0, .max_dest_rd_atomic = 1, .min_rnr_timer = 12, .ah_attr = { .is_global = 1, .dlid = remote->lid, .sl = 0, .src_path_bits = 0, .port_num = IB_PORT } };
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    attr.ah_attr.grh.hop_limit = 1;
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = { .qp_state = IBV_QPS_RTS, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = 0, .max_rd_atomic = 1 };
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

int push_memory_to_remote(struct ibv_qp *qp, struct ibv_mr *local_mr, struct shared_memory_pool *local_data, uint64_t remote_addr, uint32_t remote_rkey) {
    struct ibv_sge list = { .addr = (uintptr_t)local_data, .length = sizeof(struct shared_memory_pool), .lkey = local_mr->lkey };
    struct ibv_send_wr wr = { .wr_id = 1, .sg_list = &list, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE, .send_flags = IBV_SEND_SIGNALED, .wr.rdma.remote_addr = remote_addr, .wr.rdma.rkey = remote_rkey };
    struct ibv_send_wr *bad_wr;
    return ibv_post_send(qp, &wr, &bad_wr);
}

int main(int argc, char *argv[]) {
    int is_server = (argc == 1);
    const char *server_ip = is_server ? "192.168.100.1" : argv[1];

    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    
    struct shared_memory_pool *my_pool = calloc(1, sizeof(struct shared_memory_pool));
    struct ibv_mr *mr = ibv_reg_mr(pd, my_pool, sizeof(struct shared_memory_pool), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    struct ibv_cq *cq = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_attr = { .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC, .cap = { .max_send_wr = 10, .max_recv_wr = 10, .max_send_sge = 1, .max_recv_sge = 1 } };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, IB_PORT, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(ctx, IB_PORT, GID_INDEX, &my_gid);

    struct rdma_connection_data my_data = { .addr = (uintptr_t)mr->addr, .rkey = mr->rkey, .qp_num = qp->qp_num, .lid = port_attr.lid, .gid = my_gid };
    struct rdma_connection_data remote_data = exchange_keys_via_tcp(is_server, server_ip, &my_data);

    modify_qp_to_init(qp);
    modify_qp_to_rtr(qp, &remote_data);
    modify_qp_to_rts(qp);

    printf("\n[RDMA INFO] Shared Memory Pool Address: %p\n", (void*)my_pool);
    printf("[RDMA INFO] Remote Memory Target: 0x%lx\n", remote_data.addr);

    if (is_server) {
            printf("\n--- STEP 1: HOST GENERATING WORKLOAD ---\n");
            for(int i = 0; i < VECTOR_SIZE; i++) {
                my_pool->input_vector[i] = (float)(i + 1.1);
                my_pool->output_vector[i] = 0.0; // Clear old results
            }
            
            my_pool->job_status = 1; 
            push_memory_to_remote(qp, mr, my_pool, remote_data.addr, remote_data.rkey);

            printf("[HOST] Pushed. Waiting for VM (Check VM terminal now)...\n");
            
            // The "Volatile" check with a small sleep to force cache refresh
            while (my_pool->job_status != 2) {
                usleep(1000); 
            }

            printf("\n--- STEP 4: HOST RECEIVING RESULTS ---\n");
            for(int i = 0; i < VECTOR_SIZE; i++) {
                printf("  VM Result[%d] = %.2f\n", i, my_pool->output_vector[i]);
            }
            
            my_pool->job_status = 0; // Reset for next round
            sleep(3); 
    } else {
            if (my_pool->job_status == 1) {
                printf("\n--- STEP 2: VM PROCESSING DATA ---\n");
                for(int i = 0; i < VECTOR_SIZE; i++) {
                    my_pool->output_vector[i] = my_pool->input_vector[i] * 10.0f;
                }

                my_pool->job_status = 2; // Set status to Finished
                
                printf("[VM] Done. Pushing results back to Host...\n");
                push_memory_to_remote(qp, mr, my_pool, remote_data.addr, remote_data.rkey);

                // IMPORTANT: Wait for the RDMA card to confirm the "Push" is actually out the door
                struct ibv_wc wc;
                while (ibv_poll_cq(cq, 1, &wc) < 1); 
                
                printf("[VM] Push Confirmed. Waiting for next job...\n");
            }
            usleep(1000); 
    }
    return 0;
}