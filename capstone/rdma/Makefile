CC ?= gcc
CFLAGS ?= -Wall -O3
LIBS ?= -libverbs -lrt

all: rdma_llm rdma_pipeline

rdma_llm: rdma_llm.c
	$(CC) $(CFLAGS) rdma_llm.c -o rdma_llm $(LIBS)

rdma_pipeline: rdma_pipeline.c
	$(CC) $(CFLAGS) rdma_pipeline.c -o rdma_pipeline $(LIBS)

clean:
	rm -f rdma_llm rdma_pipeline *.o

