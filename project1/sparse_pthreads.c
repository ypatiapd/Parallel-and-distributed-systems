#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>	//include pthread
#include "mmio.h"
#include <errno.h>

//belgium_osm = 2420
//com-Youtube = 3056386
//mycielskian13 = 0
//dblp-2010 = 1676652

struct timeval startwtime, endwtime;
double seq_time;
FILE *f;

typedef struct {
    int *A;
    int *B;
    int start;
    int end;
    int nzv;
    int id;
}ThreadArgs;

struct param{
    pthread_t thread;
    ThreadArgs *args;
};

int total_sum;
int numOfWorkers=0;//number of workers alive
int numOfThreads=16;//number of threads defined by the user
int can_split=0;
int thread_id=0;
int *array_size;  //array with the remaining size of working array of each thread
pthread_mutex_t lock;

int largest(int arr[], int n);
void quicksort(int *number,int *number2,int first,int last);
int iterativeBinarySearch(int array[], int start_index, int end_index, int element);
void *find_triangles(void *arg);

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nzv;
    int  *I, *J;

    numOfThreads = atoi(argv[1]);

    if ((f = fopen(argv[2], "r")) == NULL){
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
    array_size = (int *) malloc(numOfThreads * sizeof(int));

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

    struct param Args[numOfThreads]; //arguments of thread

    gettimeofday (&startwtime, NULL);
    int cut =2*nzv/numOfThreads; //size of sub array that the thread will work on

    for(int i=0;i<numOfThreads;i++){
        Args[i].args = (ThreadArgs*)malloc (sizeof (ThreadArgs));
        Args[i].args->A=I;
        Args[i].args->B=J;
        Args[i].args->start=i*cut;
        Args[i].args->end=(i+1)*cut;
        Args[i].args->nzv=nzv;
        Args[i].args->id=i;
        array_size[i]=Args[i].args->end-Args[i].args->start;
    }
    Args[numOfThreads-1].args->end=2*nzv;

    for(int i=0;i<numOfThreads;i++){

        pthread_create (&(Args[i].thread),NULL,find_triangles,Args[i].args);
        pthread_mutex_lock (&lock);
        numOfWorkers++;
        printf(" workers=%d\n",numOfWorkers);
        pthread_mutex_unlock (&lock);
    }

    can_split=1;

    for(int j=0;j<numOfThreads;j++){
        pthread_join(Args[j].thread,NULL);

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
  int id=Arg->id;
  int z=start;
  int p=z;
  int q=0;
  int row_value=Arg->A[z];
  int col_value=Arg->B[z];
  int qsteps=0;
  int psteps=0;
  int sum=0;
  int triangles=0;
  int limit=nzv/500;// when the length of the subarray is less than this limit no more threads are created
  int waiting=0;

  struct param Args1;
  struct param Args2;

  while((Arg->A[p]==row_value)&&(p!=-1)){
      p--;
  }
  p++;

  while (z<end){

      array_size[id]=end-z;//renew the length of the remaining working interval in the global array_size array

      /*when the number of workers alive are less than the demanded from the user working threads,and the remaining
      interval of the thread that locks the mutex is higher than the limit , the sum of the splitting thread is added
      to the total sum, two new threads are generated and the splitting thread terminates.Each thread works on the half of the
      working interval of the splitting thread*/

      /*In order to split the thread with the bigger remaining working interval, the largest() function checks in the array_size array
      which thread has the bigger remaining interval, so that the work is shared efficiently*/

      if(numOfWorkers<numOfThreads){
          pthread_mutex_lock (&lock); //lock mutex to write to global variables and synchronize the creation of new threads
          array_size[id]=end-z;
          if (((end-z)==largest(array_size,numOfThreads))&&(can_split==1)&&((end-z)>limit)&&(numOfWorkers<numOfThreads)){

              array_size[id]=0;
              numOfWorkers-=1;
              printf(" workers=%d\n",numOfWorkers);
              total_sum+=sum;
              Args1.args = (ThreadArgs*)malloc (sizeof (ThreadArgs));
              Args1.args->A=Arg->A;
              Args1.args->B=Arg->B;
              Args1.args->start=z;
              Args1.args->end=z+(end-z)/2;
              Args1.args->nzv=nzv;
              Args1.args->id=Arg->id;

              Args2.args = (ThreadArgs*)malloc (sizeof (ThreadArgs));
              Args2.args->A=Arg->A;
              Args2.args->B=Arg->B;
              Args2.args->start=z+(end-z)/2;
              Args2.args->end=end;
              Args2.args->nzv=nzv;
              Args2.args->id=thread_id;

              pthread_create (&(Args1.thread),NULL,find_triangles,Args1.args);
              numOfWorkers++;
              pthread_create (&(Args2.thread),NULL,find_triangles,Args2.args);
              numOfWorkers++;
              printf(" workers=%d\n",numOfWorkers);

              waiting=1;
          }
          pthread_mutex_unlock (&lock);//unlock the mutex since the variable numOfWorkers is updated after the creation of threads.

          if(waiting==1){
              pthread_join(Args1.thread,NULL);
              pthread_join(Args2.thread,NULL);
              return 0;
          }
      }

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

  pthread_mutex_lock (&lock); //lock the mutex to write to the global variable
  total_sum+=sum;
  numOfWorkers-=1;
  printf(" workers=%d\n",numOfWorkers);
  thread_id=Arg->id;
  printf("sum=%d\n",sum);
  pthread_mutex_unlock (&lock);

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

int largest(int arr[], int n)
{
    int i;
    // Initialize maximum element
    int max = arr[0];
    // Traverse array elements
    // from second and compare
    // every element with current max
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}
