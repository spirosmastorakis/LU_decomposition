CC=gcc
MCC=mpicc
CFLAGS=-Wall -O3 
OMP=-fopenmp

all: lu_serial lu_omp lu_block_p2p lu_block_bcast lu_cyclic_p2p lu_cyclic_bcast

OBJS=utils.o
HDEPS+=%.h

lu_serial: $(OBJS) LU_serial.c
	$(CC) $(CFLAGS) $(OBJS) LU_serial.c -o lu_serial
lu_omp: $(OBJS) LU_omp.c
	$(CC) $(CFLAGS) $(OMP) $(OBJS) LU_omp.c -o lu_omp
lu_block_p2p: $(OBJS) LU_block_p2p.c
	$(MCC) $(CFLAGS) $(OBJS) LU_block_p2p.c -o lu_block_p2p
lu_block_bcast: $(OBJS) LU_block_bcast.c
	$(MCC) $(CFLAGS) $(OBJS) LU_block_bcast.c -o lu_block_bcast
lu_cyclic_p2p: $(OBJS) LU_cyclic_p2p.c
	$(MCC) $(CFLAGS) $(OBJS) LU_cyclic_p2p.c -o lu_cyclic_p2p
lu_cyclic_bcast: $(OBJS) LU_cyclic_bcast.c
	$(MCC) $(CFLAGS) $(OBJS) LU_cyclic_bcast.c -o lu_cyclic_bcast

%.o: %.c $(HDEPS)
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm lu_serial lu_omp lu_block_p2p lu_block_bcast lu_cyclic_p2p lu_cyclic_bcast utils.o

