# Parallel-and-distributed-systems

Each parallel program accepts as argument the number of desired threads and the array for calculation.

Run Example:

make clean

make all 

Serial run examples: 

./sparse_serial belgium_osm.mtx

./sparse_serial com-Youtube.mtx

./sparse_serial mycielskian13.mtx

./sparse_serial dblp-2010.mtx

./sparse_serial NACA0015.mtx


Pthreads run examples:

./sparse_pthreads 4 belgium_osm.mtx

./sparse_pthreads 4 com-Youtube.mtx

./sparse_pthreads 4 mycielskian13.mtx

./sparse_pthreads 4 dblp-2010.mtx

./sparse_pthreads 4 NACA0015.mtx
  
openMP run examples:

./sparse_openMP 4 belgium_osm.mtx

./sparse_openMP 4 com-Youtube.mtx

./sparse_openMP 4 mycielskian13.mtx

./sparse_openMP 4 dblp-2010.mtx

./sparse_openMP 4 NACA0015.mtx

cilkplus run examples:

./sparse_cilkplus 4 belgium_osm.mtx

./sparse_cilkplus 4 com-Youtube.mtx

./sparse_cilkplus 4 mycielskian13.mtx

./sparse_cilkplus 4 dblp-2010.mtx

./sparse_cilkplus 4 NACA0015.mtx


**little changes occured about the array argument for earier runing process
