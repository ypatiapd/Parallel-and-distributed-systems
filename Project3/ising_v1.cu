

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

__global__ void moment( int *array1,int *array2,int N,int L)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int sum=0;
  int row = idx /L;
  int me = array1[idx];
  int n = array1[(row)?(idx-L):(idx+L*(L-1))];
  int s = array1[(idx+L)%N];
  int e = array1[(idx + 1)%L + row * L] ;
  int w = array1[idx?((idx - 1)%L + row * L):(L-1)] ;

  sum = sum + me +n + w + s + e ;

  array2[idx]= (sum > 0) - (sum < 0);

}

int main()
{
  int *array_host, *array1_device, *array2_device ;
  int L=1000;
  int N=L*L;
  int k=40;
  int r=0;

  float ms;

  struct timeval startwtime, endwtime;
  double seq_time;
  cudaEvent_t start,stop;
  int blockSize = 500;
  int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
  srand(time(NULL));

  size_t size = N*sizeof(int);
  array_host=(int*)malloc(size);

  for (int i=0;i<N;i++) {
        r = rand() % 2;
        if(r==0)
            r=-1;
        array_host[i]=r;
  }

  cudaMalloc((void**)&array1_device, size);
  cudaMalloc((void**)&array2_device, size);
  cudaMemcpy(array1_device, array_host, size, cudaMemcpyHostToDevice);

  printf("numofblocks=%d\n",nBlocks);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  gettimeofday (&startwtime, NULL);
  for (int i=0;i<k;i++){
      //printf("z=%d\n",z);
      if(i%2==0){
          moment<<<nBlocks,blockSize>>>(array1_device,array2_device,N,L);
      }
      else{
          moment<<<nBlocks,blockSize>>>(array2_device,array1_device,N,L);
      }
  }

  cudaThreadSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  printf("time gpu =%f\n",ms);

  gettimeofday (&endwtime, NULL);
  seq_time = (endwtime.tv_sec -startwtime.tv_sec)*1000000L  +endwtime.tv_usec - startwtime.tv_usec ;


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
