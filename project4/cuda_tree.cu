#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

struct T{
    float *vp;
    float md;
    int idx;
    struct T *inner;
    struct T *outer;
};

struct T *vpTree ( float *X ,int *idx,int n,int d,int start,int threadNo,int limit);

#define SWAP(x, y) { float temp = x; x = y; y = temp; }

float quickselect(float nums[], int left, int right, int k);
float partition(float a[], int left, int right, int pIndex);
float drand ( float low, float high );

__global__ void dist( float *array,float *distances,int n,int d)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float a=0;
    float b=0;
    float sum=0;
    //printf("t idx= %d \n",threadIdx.x);
    if(idx<n){
        for(int j=0;j<d;j++){
            b=array[idx*d+j]-array[(n-1)*d+j];
            a=b*b;
            sum+=a;
        }
        distances[idx]=sum;
        //printf("idx= %d  dist= %f \n",idx,distances[idx]);
    }
    //printf("idx= %d threadid =%d \n",idx,threadIdx.x);
}
int main(int argc, char *argv[]){

    omp_set_nested(1);	//activate nested parallelism
    omp_set_dynamic(0);	//disables the dynamic adjustment of nofT
    int threadNo=atoi(argv[3]); //arithmos threads sto cilk_for , des an ginetai na allazei kai sto cilk_spawn,de nomizw
    omp_set_num_threads(threadNo);
    int p=atoi(argv[1]);
    int n= pow(2,p);
    int d =atoi(argv[2]);
    printf("n=%d  d=%d \n",n,d);
    int limit= n/2;  // limit sti dimiourgia newn threads analoga me to n
    struct timeval startwtime, endwtime;
    float seq_time;
    double sum=0;
    double ftime=0;
    struct T *t;
    int iters=1;
    for(int z=0;z<iters;z++){
        srand(time(NULL));
        float * X;
        int *idx;
        X=(float*)malloc(d*n*sizeof(float*));
        idx=(int*)malloc(n*sizeof(int*));
        for (int i=0;i<n;i++) {
            for (int j=0;j<d;j++) {
                X[i*d+j] = drand(0,1000);
            }
        }
        for( int i=0;i<n;i++){
            idx[i]=i;
        }
        gettimeofday (&startwtime, NULL);
        t=vpTree(X,idx,n,d,0,n,limit);
        gettimeofday (&endwtime, NULL);

        seq_time = (endwtime.tv_sec -startwtime.tv_sec)*1000000L  +endwtime.tv_usec - startwtime.tv_usec ;
        printf("time=%f\n",seq_time);
        if(z>0)
            sum+=seq_time;
    }
    //UNCOMENT for many iterations
    //ftime=sum/(iters);
    //printf("med_time=%lf\n",ftime);
}

struct T *vpTree ( float *X, int *idx,int n ,int d,int start,int threadNo,int limit){
    struct T *tree = (struct T*)malloc(sizeof(struct T));//dimiourgw me malloc giati alliws i metavliti pou deixnei o deiktis xanetai otan epistrepsei i sinartisi
    float median_distance;
    float *distances_dev;
    float *distances;
    float *qs_distances;
    float *x_out;
    int *outer_idxs;
    int *inner_idxs;
    distances=(float*)malloc((n)*sizeof(float*));
    qs_distances=(float*)malloc((n)*sizeof(float*));
    x_out=(float*)malloc(d*(int)(n/2)*sizeof(float*));
    outer_idxs=(int*)malloc((int)(n/2)*sizeof(int*));
    inner_idxs=(int*)malloc((int)(n/2)*sizeof(int*));
    tree->vp=(float*)malloc((d)*sizeof(float*));

    if(n==1){
        tree->md=0;
        tree->idx=idx[n-1];
        for(int i=0;i<d;i++){
            tree->vp[i]=X[(start+n-1)*d+i];
        }
        tree->inner=NULL;
        tree->outer=NULL;
    }
    if(n>=limit){

        size_t array_size = n*d*sizeof(float);
        float *array_device;
        cudaMalloc((void**)&array_device, array_size);
        cudaMemcpy(array_device, &(X[start*d]), array_size, cudaMemcpyHostToDevice);
        size_t size = n*sizeof(float);
        cudaMalloc((void**)&distances_dev, size);

        int blockSize = 500;
        int nBlocks = n/blockSize + (n%blockSize == 0?0:1);

        dist<<<nBlocks,blockSize>>>(array_device,distances_dev,n,d);
        cudaThreadSynchronize();
        cudaMemcpy(distances, distances_dev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(qs_distances, distances_dev, size, cudaMemcpyDeviceToHost);
        cudaFree(distances_dev);
        cudaFree(array_device);

        median_distance= quickselect(qs_distances, 0, n-1, n/2);//i quick select anakatanemei ton pinaka giati dia oikonomia den dimiourgei antigrafo
        printf("med_dist= %f \n",median_distance);
        tree->md=median_distance;

        for(int i=0;i<d;i++){
            tree->vp[i]=X[(start+n-1)*d+i];
        }
        tree->idx=idx[n-1]; //vazoume to deikti tou VP ston arxiko pinaka
        int index=0;

        for (int i=0;i<n;i++){ // apothikevoume ola ta megalitera se pinaka, etsi antigrafoume mono ta eswterika ta kanoume shift ston arxiko
            if(distances[i]>=median_distance){
                outer_idxs[index]=idx[i]; // apothikevoume tous deiktes twn ekswterikwn simeiwn apo to VP
                for(int j=0;j<d;j++)
                    x_out[index*d+j]=X[(start+i)*d+j];
                index++;
            }
        }
        index=0; // gia na valoume to vandage stin prwti kai na kseroume pou einai gia validation
        for (int i=0;i<n-1;i++){  //kanoume shift ta mikrotera pros ta aristera
            if(distances[i]<median_distance){
                inner_idxs[index]=idx[i]; //apothikevoume tous deiktes twn eswterikwn simeiwn apo to VP
                for(int j=0;j<d;j++)
                    X[(start+index)*d+j]=X[(start+i)*d+j];
                index++;
            }
        }
        for(int j=0;j<d;j++)
            X[(start+n/2-1)*d+j]=X[(start+n-1)*d+j]; //vazoume to vandage sto telos tou aristerou pinaka

        inner_idxs[index]=idx[n-1];// kai vazoume kai ton deikti tou vandage stous inner_idxs

        for (int i=0;i<n/2;i++){ //kai prosthetoume apo ti mesi mexri to telos tou original pinaka  ta megalitera
            for(int j=0;j<d;j++)
                X[(start+n/2+i)*d+j]=x_out[i*d+j];
        }

        /*validation*/
        float sum;
        float a;
        float b;
        for (int i=0;i<n/2;i++){
            sum=0;
            for(int j=0;j<d;j++){
                b=X[(start+i)*d+j]-X[(start+n/2-1)*d+j];
                a=b*b;
                sum+=a;
            }
            if(int(sum)>int(median_distance)){
                printf("1 errooooor iner %d dist= %f med=%f \n",i,sum,median_distance);
            }
        }
        for (int i=n/2;i<n;i++){
            sum=0;
            for(int j=0;j<d;j++){
                b=X[(start+i)*d+j]-X[(start+n/2-1)*d+j];
                a=b*b;
                sum+=a;
            }
            if(int(sum)<int(median_distance)){
                printf("1 errooooor outer %d dist=%f med=%f \n",i,sum,median_distance);

            }
        }
        free(distances); //malloc kai free mesa sto if statement
        free(qs_distances);
        free(x_out);
        free(idx);
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                  tree->inner=vpTree(X,inner_idxs,n/2,d,start,threadNo,limit); //anadromiki klisi me ta eswterika simeia
                }
                tree->outer=vpTree(X,outer_idxs,n/2,d,start+n/2,threadNo,limit); //anadromiki klisi me ta ekswterika simeia
                #pragma omp taskwait
            }
            #pragma omp barrier
        }
    }
    if((n<limit)&&(n>1)){
        float sum=0;
        float a=0;
        float b=0;
        for (int i=0;i<n;i++){
            sum=0;
            for(int j=0;j<d;j++){
                b=X[(start+i)*d+j]-X[(start+n-1)*d+j];
                a=b*b;
                sum+=a;
            }
            distances[i]=sum;
            qs_distances[i]=sum;
        }
        median_distance= quickselect(qs_distances, 0, n-1, n/2);//i quick select anakatanemei ton pinaka giati dia oikonomia den dimiourgei antigrafo

        tree->md=median_distance;
        for(int i=0;i<d;i++){
            tree->vp[i]=X[(start+n-1)*d+i];
        }
        tree->idx=idx[n-1]; //vazoume to deikti tou VP ston arxiko pinaka

        int index=0;

        for (int i=0;i<n;i++){ // apothikevoume ola ta megalitera se pinaka, etsi antigrafoume mono ta eswterika ta kanoume shift ston arxiko
            if(distances[i]>=median_distance){
                outer_idxs[index]=idx[i]; // apothikevoume tous deiktes twn ekswterikwn simeiwn apo to VP
                for(int j=0;j<d;j++)
                    x_out[index*d+j]=X[(start+i)*d+j];
                index++;
            }
        }
        index=0; // gia na valoume to vandage stin prwti kai na kseroume pou einai gia validation
        for (int i=0;i<n-1;i++){  //kanoume shift ta mikrotera pros ta aristera
            if(distances[i]<median_distance){
                inner_idxs[index]=idx[i]; //apothikevoume tous deiktes twn eswterikwn simeiwn apo to VP
                for(int j=0;j<d;j++)
                    X[(start+index)*d+j]=X[(start+i)*d+j];
                index++;
            }
        }
        for(int j=0;j<d;j++)
            X[(start+n/2-1)*d+j]=X[(start+n-1)*d+j]; //vazoume to vandage sto telos tou aristerou pinaka

        inner_idxs[index]=idx[n-1];// kai vazoume kai ton deikti tou vandage stous inner_idxs

        for (int i=0;i<n/2;i++){ //kai prosthetoume apo ti mesi mexri to telos tou original pinaka  ta megalitera
            for(int j=0;j<d;j++)
                X[(start+n/2+i)*d+j]=x_out[i*d+j];
        }

        /*validation*/

        for (int i=0;i<n/2;i++){
            sum=0;
            float b;
            float a;
            for(int j=0;j<d;j++){
                b=X[(start+i)*d+j]-X[(start+n/2-1)*d+j];
                a=b*b;
                sum+=a;
            }
            if(sum>median_distance){
                printf("2 errooooor iner %d dist= %f med =%f n= %d \n",i,sum,median_distance,n );
            }
        }
        for (int i=n/2;i<n;i++){
            sum=0;
            float a;
            float b;
            for(int j=0;j<d;j++){
                b=X[(start+i)*d+j]-X[(start+n/2-1)*d+j];
                a=b*b;
                sum+=a;
            }
            if(sum<median_distance){
                printf("2 errooooor outer %d dist=%f med= %f n=%d  \n",i,sum,median_distance,n);
            }
        }
        free(distances);
        free(qs_distances);
        free(x_out);
        free(idx);
        tree->inner=vpTree(X,inner_idxs,n/2,d,start,threadNo,limit); //anadromiki klisi me ta eswterika simeia
        tree->outer=vpTree(X,outer_idxs,n/2,d,start+n/2,threadNo,limit); //anadromiki klisi me ta ekswterika simeia
    }
    return tree;
}

float drand ( float low, float high )
{
    return ( (float)rand() * ( high - low ) ) / (float)RAND_MAX + low;
}

float quickselect(float nums[], int left, int right, int k)
{
    // If the array contains only one element, return that element
    if (left == right) {
        return nums[left];
    }

    // select `pIndex` between left and right
    int pIndex = left + rand() % (right - left + 1);

    pIndex = partition(nums, left, right, pIndex);

    // The pivot is in its final sorted position
    if (k == pIndex) {
        return nums[k];
    }

    // if `k` is less than the pivot index
    else if (k < pIndex) {
        return quickselect(nums, left, pIndex - 1, k);
    }

    // if `k` is more than the pivot index
    else {
        return quickselect(nums, pIndex + 1, right, k);
    }
}

float partition(float a[], int left, int right, int pIndex)
{
    // pick `pIndex` as a pivot from the array
    float pivot = a[pIndex];

    // Move pivot to end
    SWAP(a[pIndex], a[right]);

    // elements less than the pivot will be pushed to the left of `pIndex`;
    // elements more than the pivot will be pushed to the right of `pIndex`;
    // equal elements can go either way
    pIndex = left;

    // each time we find an element less than or equal to the pivot, `pIndex`
    // is incremented, and that element would be placed before the pivot.
    for (int i = left; i < right; i++)
    {
        if (a[i] <= pivot)
        {
            SWAP(a[i], a[pIndex]);
            pIndex++;
        }
    }

    // move pivot to its final place
    SWAP(a[pIndex], a[right]);

    // return `pIndex` (index of the pivot element)
    return pIndex;
}
