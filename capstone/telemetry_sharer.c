#include <unistd.h>
#include <rdma_arbitrator.c>

void start_telemetry_heartbeat(struct ibv_qp *qp, struct ibv_mr *local_mr, uint64_t remote_addr, uint32_t remote_rkey) {
    
    // Point our struct to the registered memory region
    struct telemetry_data *my_telemetry = (struct telemetry_data *)local_mr->addr;

    printf("Starting Telemetry Heartbeat (Pushing data every 10ms)...\n");

    while (1) {
        // In a real scenario, you'd parse /proc/stat here. 
        // We'll mock it for the architecture test.
        my_telemetry->host_free_cpu_pct = 85; 
        my_telemetry->host_free_ram_mb = 16384;
        my_telemetry->inference_queue_len = 0;

        // Fire it across the Soft-RoCE bridge
        push_telemetry_to_remote(qp, local_mr, my_telemetry, remote_addr, remote_rkey);

        // Sleep for 10 milliseconds
        usleep(10000); 
    }
}