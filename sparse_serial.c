#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mmio.h"
#include <errno.h>

/*Triangles in each graph*/
//belgium_osm = 2420
//com-Youtube = 3056386
//mycielskian13 = 0
//dblp-2010 = 1676652


struct timeval startwtime, endwtime;
double seq_time;
FILE *f;

void quicksort(int *number,int *number2,int first,int last);
int iterativeBinarySearch(int array[], int start_index, int end_index, int element);
int find_triangles(int *A, int *B , int nz);

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nzv; //number of rows and columns in the main array , and non zero values
    int *I, *J ;

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

    /*Triangle counting function*/
    find_triangles(I, J ,nzv);

    gettimeofday (&endwtime, NULL);

    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
  		      + endwtime.tv_sec - startwtime.tv_sec);

    printf("time=%f\n",seq_time);

}
/*This function calculates the number of triangles in the sparse graph. It takes as arguments one vector A with the indexes
 of the rows of the sparse graph that have a connection and one vector B with all the indexes of the columns of the sparse graph
that have connection with the rows of the A vector .The two vectors are sorted by the first vector(CSR) .
Because of the fact that the graph is symmetric the connections of the rows are the same with the connections of the columns ,
so we can search for each pair of A and B that is connected in the sparse graph, the values of the nodes
in vector B in the sorted vector A and then find their common connections in the vector B.
We check iteratively through the main loop that runs for as many iterations as the number of the connections of the graph ,
and for each pair of A and B vector we search the value of the B vector to the sorted A vector and then for all the connections of the two nodes ,
we search at B vector for their common connections .These common connections imply the existance of triangle
between the two examined nodes and their common connection.This proceedure takes place for all the pairs of A and B vector
and the final sum of the common connections divided by 6 ( with 2 because the array is symmetric and with 3 because each triangle has 3 edges)
gives the number of the triangles of the array */
int find_triangles(int *A, int *B,int nz){

  int row_value=0;  //value of current row of the sparse graph examined
  int col_value=0;  //value of  current column of the sparse graph examined
  int z=0;  //iterator for the basic loop
  int p=0;  //iterator that moves sequentialy through the values of the A vector
  int q=0;  //iterator that points at the values of B vector found in A vector
  int qsteps=0;  //variable that counts the number of times one value exists in the sorted vector (number of connections of each node)
  int psteps=0;  //same for the p iterator
  int sum=0;   //sum of common values
  int triangles=0;  //triangles in the graph

  while (z<2*nz){

      if((A[z]==row_value)&&(z!=0)){ // if the number of row of the main array hasn't changed ,
                                    //p iterator goes back at the position with the first connection of the current node for a new comparison
          p-=psteps;
      }
      q=p;    //set q value ewual with p for more efficient search
      psteps=0;
      qsteps=0;
      row_value=A[z];   //renew current row value
      col_value=B[z];   //renew current col value

      /*calculate the number of connections of the current row , equals psteps value*/
      while(A[p]==row_value){
          p++;
          psteps++;
      }
      p-=psteps;

      int a[psteps];
      /*for all the connections of the current row , we save the values of connections (columns)
      to a temporary array for saving time from reading from memory*/
      for(int i=0;i<psteps;i++){
          a[i]=B[p];
          p++;
      }
      /*search of the current value of B vector to vector A. We start of p value,
      and because the main array is symmetric , if the row value < col value we search above
      the p value, otherwise we search below the p value*/
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
      /*When the column value is found at A vector ,we calculate the
      number of connections in the vector , equals qsteps value */
      while(A[q]==col_value){
          q++;
          qsteps++;
      }
      q-=qsteps;

      int b[qsteps];
      /*for all the connections of the node , we save the values of connections
      to a temporary array for saving time from reading from memory*/
      for(int i=0;i<qsteps;i++){
          b[i]=B[q];
          q++;
      }
      /*for the two vectors of connections of the two examined nodes, we compare for common values.
      Each common value found, implies a triangle , because both nodes have connection with a third one */
      for(int i=0;i<psteps;i++){
          for(int j=0;j<qsteps;j++){
              if(a[i]==b[j]){
                  sum++; //increase the sum of connections
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
