#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>



struct T{
    double *vp;
    double md;
    int idx;
    struct T *inner;
    struct T *outer;
};

struct T *vpTree ( double *X ,int *idx,int n,int d,int start,int threadNo,int limit);

#define SWAP(x, y) { double temp = x; x = y; y = temp; }

double quickselect(double nums[], int left, int right, int k);
double partition(double a[], int left, int right, int pIndex);
void printLeafs(struct T *t);
double drand ( double low, double high );

int main(int argc, char *argv[]){

    omp_set_nested(1);	//activate nested parallelism
    omp_set_dynamic(0);	//disables the dynamic adjustment of nofT
    int threadNo=atoi(argv[3]);
    omp_set_num_threads(threadNo);	//set number of threads
    int p=atoi(argv[1]);
    int n= pow(2,p);
    int d =atoi(argv[2]);
    printf("n=%d  d=%d \n",n,d);
    int limit= n/2;  // limit sti dimiourgia newn threads analoga me to n
    struct timeval startwtime, endwtime;
    double seq_time;
    double sum=0;
    double ftime=0;
    struct T *t;
    int iters=1;
    for(int z=0;z<iters;z++){
        srand(time(NULL));
        double * X;
        int *idx;
        X=(double*)malloc(d*n*sizeof(double*));
        idx=(int*)malloc(n*sizeof(int*));

        for (int i=0;i<n;i++) {
            for (int j=0;j<d;j++) {
                X[i*d+j] = drand(0,1000);
            }
        }
        /*for (int i=0;i<n;i++) {
            for (int j=0;j<d;j++) {
                printf("%f ",  X[i*d+j]);
            }
            printf(" \n");
        }*/

        for( int i=0;i<n;i++){
            idx[i]=i;
        }

        gettimeofday (&startwtime, NULL);
        t=vpTree(X,idx,n,d,0,threadNo,limit);
        gettimeofday (&endwtime, NULL);

        seq_time = (endwtime.tv_sec -startwtime.tv_sec)*1000000L  +endwtime.tv_usec - startwtime.tv_usec ;
        printf("time=%f\n",seq_time);

        sum+=seq_time;

    }
    //UNCOMENT for many iterations
    //ftime=sum/(iters);
    //printf("med_time=%lf\n",ftime);
}

struct T *vpTree ( double *X, int *idx,int n ,int d,int start,int threadNo,int limit){
    struct T *tree =(struct T*) malloc(sizeof(struct T));//dimiourgw me malloc giati alliws i metavliti pou deixnei o deiktis xanetai otan epistrepsei i sinartisi
    int block=n/threadNo;
    double median_distance;
    double *distances;
    double *qs_distances;
    double *x_out;
    int *outer_idxs;
    int *inner_idxs;
    distances=(double*)malloc((n)*sizeof(double*));
    qs_distances=(double*)malloc((n)*sizeof(double*));
    x_out=(double*)malloc(d*(int)(n/2)*sizeof(double*));
    outer_idxs=(int*)malloc((int)(n/2)*sizeof(int*));
    inner_idxs=(int*)malloc((int)(n/2)*sizeof(int*));
    tree->vp=(double*)malloc((d)*sizeof(double*));

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
        #pragma omp parallel
        {
            int q;
            q=omp_get_thread_num();
            double sum=0;
            double a=0;
            for(int i=q*block;i< q*block + block ;i++){
                sum=0;
                for(int j=0;j<d;j++){
                    a=pow(X[(start+i)*d+j]-X[(start+n-1)*d+j],2);
                    sum+=a;
                }
                distances[i]=sqrt(sum);
                qs_distances[i]=sqrt(sum);
            }
            #pragma omp barrier
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
                outer_idxs[index]=idx[i] ;// apothikevoume tous deiktes twn ekswterikwn simeiwn apo to VP
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
        double sum;
        double a;
        for (int i=0;i<n/2;i++){
            sum=0;
            for(int j=0;j<d;j++){
                a=pow(X[(start+i)*d+j]-X[(start+n/2-1)*d+j],2);
                sum+=a;
            }
            if(sqrt(sum)>=median_distance){
                printf("errooooor iner %d dist= %f  \n",i,sqrt(sum) );
            }
        }
        for (int i=n/2;i<n;i++){
            sum=0;
            for(int j=0;j<d;j++){
                a=pow(X[(start+i)*d+j]-X[(start+n/2-1)*d+j],2);
                sum+=a;
            }
            if(sqrt(sum)<median_distance){
                printf("errooooor outer %d dist=%f \n",i,sqrt(sum));
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
        double sum=0;
        double a=0;

        for (int i=0;i<n;i++){
            sum=0;
            for(int j=0;j<d;j++){
                a=pow(X[(start+i)*d+j]-X[(start+n-1)*d+j],2);
                sum+=a;
            }
            distances[i]=sqrt(sum);
            qs_distances[i]=sqrt(sum);//des an boreis na to kaneis alliws, na epistrefei ta indexes i quickselect
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
            for(int j=0;j<d;j++){
                a=pow(X[(start+i)*d+j]-X[(start+n/2-1)*d+j],2);
                sum+=a;
            }
            if(sqrt(sum)>=median_distance){
                printf("errooooor iner %d dist= %f  \n",i,sqrt(sum) );
            }
        }
        for (int i=n/2;i<n;i++){
            sum=0;
            for(int j=0;j<d;j++){
                a=pow(X[(start+i)*d+j]-X[(start+n/2-1)*d+j],2);
                sum+=a;
            }
            if(sqrt(sum)<median_distance){
                printf("errooooor outer %d dist=%f \n",i,sqrt(sum));
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

double quickselect(double nums[], int left, int right, int k)
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

double partition(double a[], int left, int right, int pIndex)
{
    // pick `pIndex` as a pivot from the array
    double pivot = a[pIndex];

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

void printLeafs(struct T *t)
{
    // if node is null, return
    //if (!root)
    //    return;

    // if node is leaf node, print its data
    if (!(t->inner) && !(t->outer))
    {
        printf("idx= %d\n",t->idx);
        return;
    }

    // if left child exists, check for leaf
    // recursively
    if (t->inner)
       printLeafs(t->inner);

    // if right child exists, check for leaf
    // recursively
    if (t->outer)
       printLeafs(t->outer);
}

double drand ( double low, double high )
{
    return ( (double)rand() * ( high - low ) ) / (double)RAND_MAX + low;
}
