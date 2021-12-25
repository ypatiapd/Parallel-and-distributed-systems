CC=gcc
MPICC=mpicc
CILKCC=/usr/local/OpenCilk-9.0.1-Linux/bin/clang
CFLAGS=-O3

default: all

sparse_serial:
	$(CC) $(CFLAGS) -o sparse_serial sparse_serial.c mmio.o

sparse_cilkplus:
	$(CC) $(CFLAGS) -o sparse_cilkplus sparse_cilkplus.c mmio.o -fcilkplus

sparse_openMP:
	$(CC) $(CFLAGS) -o sparse_openMP sparse_openMP.c mmio.o -fopenmp

sparse_pthreads:
	$(CC) $(CFLAGS) -o sparse_pthreads sparse_pthreads.c mmio.o -lpthread

all: sparse_serial sparse_cilkplus sparse_openMP sparse_pthreads 
clean:
	rm  sparse_serial sparse_cilkplus sparse_openMP sparse_pthreads 
