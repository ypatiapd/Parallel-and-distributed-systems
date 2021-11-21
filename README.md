# Parallel-and-distributed-systems


Each parallel program accepts as argument the number of desired threads.

Run Example:

make clean
make all 

./sparse_serial \n
./sparse_pthreads 8 \n
./sparse_cilkplus 16  \n
./sparse_openMP 32   \n

