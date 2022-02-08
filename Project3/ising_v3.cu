

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

__global__ void moment( int *array1,int *array2,int N,int L,int b,int nBlocks,int blockSize)
{
    int idx= blockIdx.x*blockDim.x*b + threadIdx.x*b;
    __shared__ int array_block[1500];
    int row = idx /L;   //row of the array in global memory
    int shared_idx=threadIdx.x*b;   //index of point in shared memory
    int left,right;                 // left and right neighbours of the edge points  in shared memory (we need only two variables because we pass a Line of the main array or less)
    int shared_L=blockSize*b;       // line length of the shared memory array

    for (int i=0;i<b;i++){
        row = (idx+i) /L;
        array_block[shared_idx+i]=array1[idx+i];//every thread copies to shared memory its points from global (from 1 to b)
        array_block[shared_idx+i+ shared_L]=array1[(idx+i+L)%N]; //for these points, each thread copies the south neighbours to the second line of the shared memory array
        array_block[shared_idx+i+ 2*shared_L]=array1[(row)?(idx-L+i):(idx+L*(L-1)+i)];  // also for these points, each thread copies the north neighbours to the third line of the shared memory array
    }
    
    //if the point is at the left edge of the points vector then copy to left variable the left neighbour
    if(shared_idx==0){
        left=array1[(idx)?((idx - 1)%L + row * L):(L-1)] ;
    }
    
    //if the point is at the right edge of the points vector then copy to right variable the right neighbour
    if(shared_idx+b==shared_L){
        row = (idx+b-1) /L;
        right=array1[(idx+b)%L + row * L] ;
    }

    __syncthreads();

    for (int i =0;i<b;i++){
        int sum=0;
        int me = array_block[shared_idx+i];
        int n = array_block[shared_idx+i+2*shared_L];
        int s = array_block[shared_idx+i+ shared_L];
        int e = (shared_L-shared_idx-i-1)?(array_block[shared_idx+i + 1]):(right) ;
        int w = (shared_idx+i)?( array_block[shared_idx+i - 1]):(left) ;
        sum = sum + me +n + w + s + e ;

        array2[idx+i]= (sum > 0) - (sum < 0);
    }

int main()
{
  int *array_host, *array1_device, *array2_device ;
  int L=2000;
  int N=L*L;
  int k=40;
  int b=5;

  int r=0;
  int i,j;

  int blockSize=500/b;
  int nBlocks = (N/blockSize)/b + (N%(blockSize*b)== 0?0:1);

  cudaEvent_t start,stop;
  float ms;

  struct timeval startwtime, endwtime;
  double seq_time;

  srand(time(NULL));
  size_t size = N*sizeof(int);
  array_host=(int*)malloc(size);

  for (i=0;i<N;i++) {
        r = rand() % 2;
        if(r==0)
            r=-1;
        array_host[i]=r;
  }


  cudaMalloc((void**)&array1_device, size);
  cudaMalloc((void**)&array2_device, size);
  cudaMemcpy(array1_device, array_host, size, cudaMemcpyHostToDevice);

  printf("num %d   \n",nBlocks);

  gettimeofday (&startwtime, NULL);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (i=0;i<k;i++){
      if(i%2==0){
          moment<<<nBlocks,blockSize,sharedBytes>>>(array1_device,array2_device,N,L,b,nBlocks,blockSize);
      }
      else{
          moment<<<nBlocks,blockSize,sharedBytes>>>(array2_device,array1_device,N,L,b,nBlocks,blockSize);
      }
  }
  cudaThreadSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);

  printf("time=%f\n",seq_time);
  printf("time gpu =%f\n",ms);
  if(k%2==0){
      cudaMemcpy(array_host, array1_device, size, cudaMemcpyDeviceToHost);
  }
  else{
      cudaMemcpy(array_host, array2_device, size, cudaMemcpyDeviceToHost);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(array1_device);
  cudaFree(array2_device);
  return 0;
}
