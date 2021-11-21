#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mmio.h"
#include <errno.h>


//belgium_osm = 2420
//com-Youtube = 3056386
//mycielskian13 = 0
//dblp-2010 = 1676652


struct timeval startwtime, endwtime;
double seq_time;
FILE *f;

void quicksort(int *number,int *number2,int first,int last);
int iterativeBinarySearch(int array[], int start_index, int end_index, int element);
int find_triangles(int *A, int *B , int nz, int M);

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nzv;
    int i, *I, *J ;

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

    for (int i=0; i<nzv; i++)
    {
        fscanf(f, "%d %d\n", &J[i], &I[i]);
        I[i]--;
        J[i]--;
    }

    fclose(f);

    for (int j=0; j<nzv; j++){
        I[nzv+j]=J[j];
        J[nzv+j]=I[j];
    }

    quicksort(I,J,0,2*nzv-1);

    gettimeofday (&startwtime, NULL);


    find_triangles(I, J ,nzv ,M);

    gettimeofday (&endwtime, NULL);

    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
  		      + endwtime.tv_sec - startwtime.tv_sec);

    printf("time=%f\n",seq_time);

}

int find_triangles(int *A, int *B,int nz,int M){

  int row_value=0;
  int col_value=0;
  int z=0;
  int p=0;
  int q=0;
  int qsteps=0;
  int psteps=0;
  int sum=0;
  int triangles=0;
  int same_row=0;


  while (z<2*nz){


      if((A[z]==row_value)&&(z!=0)){
          p-=psteps;
      }
      q=p;
      psteps=0;
      qsteps=0;
      row_value=A[z];
      col_value=B[z];

      while(A[p]==row_value){
          p++;
          psteps++;
      }
      p-=psteps;

      int a[psteps];

      for(int i=0;i<psteps;i++){
          a[i]=B[p];
          p++;
      }

      if(row_value<col_value){
          q=iterativeBinarySearch(A, p, 2*nz-1, col_value);
          while(A[q]==col_value){
              q--;
          }
          q++;
      }
      else{
          q=iterativeBinarySearch(A, 0, p, col_value);
          while((A[q]==col_value)&&(q!=-1)){
              q--;
          }
          q++;
      }
      while(A[q]==col_value){
          q++;
          qsteps++;
      }
      q-=qsteps;


      int b[qsteps];

      for(int i=0;i<qsteps;i++){
          b[i]=B[q];
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
  triangles=sum/6;
  printf("triangles=%d\n",triangles);
  printf("sum=%d\n",sum);
  return triangles;
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
