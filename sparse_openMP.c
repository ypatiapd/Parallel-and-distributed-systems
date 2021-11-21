#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>	//include for openMP
#include "mmio.h"
#include <errno.h>


struct timeval startwtime, endwtime;
double seq_time;
FILE *f;

typedef struct {
    int *A;
    int *B;
    int start;
    int end;
    int nzv;
}ThreadArgs;

struct param{
    pthread_t thread;
    ThreadArgs *args;
};

int total_sum;
int numOfThreads;


void quicksort(int *number,int *number2,int first,int last);
int iterativeBinarySearch(int array[], int start_index, int end_index, int element);
void *find_triangles(void *arg);

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nzv;
    int  *I, *J ;

    numOfThreads= atoi(argv[1]);;

    struct param Args[numOfThreads];

    if ((f = fopen("belgium_osm.mtx", "r")) == NULL){
        printf("NULL pointer\n");
        perror("fopen");

    }
    if (mm_read_banner(f, &matcode) != 0)
    {
          printf("Could not process Matrix Market banner.\n");
          exit(1);
    }
      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N ,&nzv)) !=0){
        printf("ERROOORRR");
        exit(1);
    }
    printf("M=%d\n",M);
    printf("Ν=%d\n",N);
    /* reseve memory for matrices */

    I = (int *) malloc(2*nzv * sizeof(int));
    J = (int *) malloc(2*nzv * sizeof(int));

    /*Store the rows at array I and columns at array J*/
    for (int i=0; i<nzv; i++)
    {
        fscanf(f, "%d %d\n", &J[i], &I[i]);
        I[i]--;
        J[i]--;
    }

    fclose(f);

    /*Convert from triagonal to symmetric quadratic*/
    for (int j=0; j<nzv; j++){
        I[nzv+j]=J[j];
        J[nzv+j]=I[j];
    }

    /*Sort row values of I following CSR .J values follow the sorting of I*/
    quicksort(I,J,0,2*nzv-1);

    gettimeofday (&startwtime, NULL);
    int cut =2*nzv/numOfThreads; //size of sub array that the thread will work on

    for(int i=0;i<numOfThreads;i++){
        Args[i].args = (ThreadArgs*)malloc (sizeof (ThreadArgs));
        Args[i].args->A=I;
        Args[i].args->B=J;
        Args[i].args->start=i*cut;
        Args[i].args->end=(i+1)*cut;
        Args[i].args->nzv=nzv;
    }
    Args[numOfThreads-1].args->end=2*nzv;

    omp_set_num_threads(numOfThreads);	//set number of threads
    omp_set_nested(1);	//activate nested parallelism
    omp_set_dynamic(0);	//disables the dynamic adjustment of nofT

    #pragma omp parallel
    {
      #pragma omp for
        for(int i=0;i<numOfThreads;i++){
              #pragma omp task
              find_triangles(Args[i].args);

        }
    }
    for(int i=0;i<numOfThreads;i++){

        #pragma omp taskwait
    }
    gettimeofday (&endwtime, NULL);

    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);

    printf("time=%f\n",seq_time);
    printf("total_sum=%d\n",total_sum);
    int triangles =total_sum/6;
    printf("triangles=%d\n",triangles);

}
//commands for the basic function find_triangles at the sparse_serial file
void *find_triangles(void *arg){

  ThreadArgs *Arg;
  Arg=arg ;
  int start=Arg->start;
  int end=Arg->end;
  int nzv=Arg->nzv;
  int z=start;
  int p=z;
  int q=0;
  int row_value=Arg->A[z];
  int col_value=Arg->B[z];
  int qsteps=0;
  int psteps=0;
  int sum=0;
  int triangles=0;

  while((Arg->A[p]==row_value)&&(p!=-1)){
      p--;
  }
  p++;

  while (z<end){

      if((Arg->A[z]==row_value)&&(z!=start)){
          p-=psteps;
      }
      q=p;
      psteps=0;
      qsteps=0;
      row_value=Arg->A[z];
      col_value=Arg->B[z];

      while(Arg->A[p]==row_value){
          p++;
          psteps++;
      }
      p-=psteps;

      int a[psteps];

      for(int i=0;i<psteps;i++){
          a[i]=Arg->B[p];
          p++;
      }
      if(row_value<col_value){
          q=iterativeBinarySearch(Arg->A, p, 2*nzv-1, col_value);
          while(Arg->A[q]==col_value){
              q--;
          }
          q++;
      }
      else{
          q=iterativeBinarySearch(Arg->A, 0, p, col_value);
          while((Arg->A[q]==col_value)&&(q!=-1)){
              q--;
          }
          q++;
      }
      while(Arg->A[q]==col_value){
          q++;
          qsteps++;
      }
      q-=qsteps;

      int b[qsteps];

      for(int i=0;i<qsteps;i++){
          b[i]=Arg->B[q];
          q++;
      }
      for(int i=0;i<psteps;i++){
          for(int j=0;j<qsteps;j++){
              if(a[i]==b[j]){
                  sum++;
              }
          }
      }
      z++;
  }

  total_sum+=sum;
  printf("sum=%d\n",sum);

}

void quicksort(int *number,int *number2,int first,int last){
    int i, j, pivot, temp,temp2;
    if(first<last){
        pivot=first;
        i=first;
        j=last;
        while(i<j){
            while(number[i]<=number[pivot]&&i<last)
            i++;
            while(number[j]>number[pivot])
            j--;
            if(i<j){
            temp=number[i];
            temp2=number2[i];
            number[i]=number[j];
            number2[i]=number2[j];
            number[j]=temp;
            number2[j]=temp2;
            }
        }
        temp=number[pivot];
        temp2=number2[pivot];
        number[pivot]=number[j];
        number2[pivot]=number2[j];
        number[j]=temp;
        number2[j]=temp2;
        quicksort(number,number2,first,j-1);
        quicksort(number,number2,j+1,last);
    }
}

int iterativeBinarySearch(int array[], int start_index, int end_index, int element){
   while (start_index <= end_index){
      int middle = start_index + (end_index- start_index )/2;
      if (array[middle] == element)
         return middle;
      if (array[middle] < element)
         start_index = middle + 1;
      else
         end_index = middle - 1;
   }
   return -1;
}
