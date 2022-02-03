CC=gcc
CFLAGS=-O3

default: all

ising_serial:
	$(CC) $(CFLAGS) -o ising_serial ising_serial.c -lm

all: ising_serial 

clean:
	rm  ising_serial 
