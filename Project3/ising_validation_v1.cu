

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void find_moment ( int ** A, int ** B ,int L );

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

int main(int argc, char *argv[]){

    int ** I;
    int ** J;
    int L = 2000;
    int N= L*L;
    int k = 40;
    int r;

    int blockSize = 500;
    int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
    int *array_host, *array1_device, *array2_device ;
    cudaEvent_t start,stop;

    struct timeval startwtime, endwtime;
    double seq_time;

    I=(int**)malloc(L*sizeof(int*));
    J=(int**)malloc(L*sizeof(int*));
    for (int i=0;i<L;i++){
        I[i]=(int*)malloc(L*sizeof(int));
        J[i]=(int*)malloc(L*sizeof(int));
    }

    for (int i=0;i<L;i++) {
        for (int j=0;j<L;j++) {
            r = rand() % 2;
            if(r==0)r=-1;
            I[i][j]=r;
        }
    }

    size_t size = N*sizeof(int);
    array_host=(int*)malloc(size);

    for(int i=0;i<L;i++){
        for(int j=0;j<L;j++)
            array_host[i*L+j]=I[i][j];
    }

    for (int z=0;z<k;z++){
        if(z%2==0){
            find_moment(I,J,L);
        }
        else{
            find_moment(J,I,L);
        }
    }

    printf("Final array serial \n");
    /*if(k%2==0){
        for(int i=0;i<L;i++){
            for(int j=0;j<L;j++)
                printf("%d ",I[i][j]);
            printf("\n");
        }
    }
    else{
        for(int i=0;i<L;i++){
            for(int j=0;j<L;j++)
                printf("%d ",J[i][j]);
            printf("\n");
        }
    }*/

    // ---------------------------------end of serial ---------------------------------------------------------------------------


    cudaMalloc((void**)&array1_device, size);
    cudaMalloc((void**)&array2_device, size);
    cudaMemcpy(array1_device, array_host, size, cudaMemcpyHostToDevice);

    printf("numofblocks=%d\n",nBlocks);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float ms;

    gettimeofday (&startwtime, NULL);

    for (int i=0;i<k;i++){
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

    int counter=0;
    for(int  i=0;i<L;i++){
        for(int j=0;j<L;j++)
            if(I[i][j]!=array_host[i*L+j]){
                counter++;
            }
    }

    printf("counter= %d ",counter);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(array1_device);
    cudaFree(array2_device);
    return 0;
}

void find_moment ( int ** A, int ** B ,int L) {
    int sum=0;
    for (int i=0;i<L;i++) {
        for (int j=0;j<L;j++) {
            sum+=A[i][j];
            sum+=A[(i+1)%L][j];
            sum+=A[i?(i-1):(L-1)][j];
            sum+=A[i][(j+1)%L];
            sum+=A[i][j?(j-1):(L-1)];

            if(sum>0)
                B[i][j]=1;
            else
                B[i][j]=-1;
            sum=0;
        }
    }
    return ;
}
