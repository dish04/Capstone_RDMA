savedcmd_rdma_arbitrator.mod := printf '%s\n'   rdma_arbitrator.o | awk '!x[$$0]++ { print("./"$$0) }' > rdma_arbitrator.mod
