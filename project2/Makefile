CC=gcc
MPICC=mpicc
CFLAGS=-O3

default: all

main_program:
	$(MPICC) $(CFLAGS) -o main_program main_program.c -lm

distributeByMedian:
	$(MPICC) $(CFLAGS) -o distributeByMedian distributeByMedian.c -lm

all: distributeByMedian main_program

clean:
	rm  distributeByMedian main_program
