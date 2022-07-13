CC=gcc
MPICC=mpicc
CFLAGS=-O3

default: all

knn_serial:
	$(CC) $(CFLAGS) -o knn_serial knn_serial.c -lm

knn:
	$(CC) $(CFLAGS) -o knn knn.c -fcilkplus -fopenmp -lm

openmp_tree:
	$(CC) $(CFLAGS) -o openmp_tree openmp_tree.c -fopenmp -lm


all:  knn knn_serial openmp_tree

clean:
	rm  knn knn_serial openmp_tree
