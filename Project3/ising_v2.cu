

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

__global__ void moment( int *array1,int *array2,int N,int L,int b)
{
  int numOfThreads=N/b;
  int idx= blockIdx.x*blockDim.x*b + threadIdx.x*b;

  for (int i =0;i<b;i++){
      int sum=0;
      int row = (idx+i) /L;
      int me = array1[idx+i];
      int n = array1[(row)?(idx+i-L):(idx+i+L*(L-1))];
      int s = array1[(idx+i+L)%N];
      int e = array1[(idx+i + 1)%L + row * L] ;
      int w = array1[(idx+i)?((idx+i - 1)%L + row * L):(L-1)] ;
      sum = sum + me +n + w + s + e ;

      array2[idx+i]= (sum > 0) - (sum < 0);
  }
}

int main()
{
  int *array_host, *array1_device, *array2_device ;
  int L=4000;
  int N=L*L;
  int k=40;
  int b=2;
  int r=0;

  int blockSize=500;
  int nBlocks = (N/blockSize)/b + (N%blockSize == 0?0:1);

  struct timeval startwtime, endwtime;
  double seq_time;
  cudaEvent_t start,stop;
  float ms;
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
          moment<<<nBlocks,blockSize>>>(array1_device,array2_device,N,L,b);
      }
      else{
          moment<<<nBlocks,blockSize>>>(array2_device,array1_device,N,L,b);
      }
  }
  cudaThreadSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  gettimeofday (&endwtime, NULL);

  seq_time = (endwtime.tv_sec -startwtime.tv_sec)*1000000L  +endwtime.tv_usec - startwtime.tv_usec ;

  printf("time gpu =%f\n",ms);
  printf("time=%f\n",seq_time);

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
