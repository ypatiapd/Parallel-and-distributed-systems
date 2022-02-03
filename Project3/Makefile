CC=gcc
CFLAGS=-O3

default: all

ising_serial:
	$(CC) $(CFLAGS) -o ising_serial ising_serial.c -lm

v1_test:
	$(CC) $(CFLAGS) -o v1_test v1_test.c -lm

all: ising_serial v1_test

clean:
	rm  ising_serial v1_test
