#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#define SWAP(x, y) { double temp = x; x = y; y = temp; }

double quickselect(double nums[], int left, int right, int k);
double partition(double a[], int left, int right, int pIndex);

int main(int argc, char *argv[])
{
    int SelfTID,NumTasks, t ,data;
    MPI_Status mpistat;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&SelfTID);
    printf("hello from %i of %i tasks\n",SelfTID,NumTasks);

    int n;
    int d;
    int trash=0;
    FILE *ptr;
    int z=0;
    ptr = fopen("file.bin","rb");  // r for read, b for binary
    if (ptr == NULL){
        printf("no file \n ");
        return -1;
    }
    z=fread(&n,sizeof(n),1,ptr); // read 10 bytes to our buffer
    z=fread(&trash,sizeof(trash),1,ptr); // read 10 bytes to our buffer
    z=fread(&d,sizeof(d),1,ptr); // read 10 bytes to our buffer
    z=fread(&trash,sizeof(trash),1,ptr); // read 10 bytes to our buffer
    printf("M= %d\n ", n);
    printf("N= %d\n ", d);

    double array[n][d];

    for (int i=0;i<n;i++)
        z=fread(&(array[i]),sizeof(array[i]),1,ptr); // read 10 bytes to our buffer
    fclose(ptr);

    for (int i =0; i<n;i++){
        printf("bin= %lf\n ", array[i][19]);
    }

    /*for (int i =0; i< n;i++){
        for(int j=0;j<d;j++){
            array[i][j]=rand() % 100;
        }
    }*/
    /*for (int i =0; i<n;i++){
        printf("%d ",array[i][0]);
        printf("%d ",array[i][1]);
        printf("%d\n",array[i][2]);
    }*/
    int sub_n=n/NumTasks;
    double pivot[d];
    double distances[sub_n];
    double all_distances[n];
    double median_distance=0;
    int choose=0;
    /*choose and announce pivot ,receive pivot from the other processes*/
    if(SelfTID == 0){
        choose=rand() % n;
        for(int i=0;i<d;i++){
            pivot[i]=array[choose][i];
        }
        //printf("pivot= %d %d %d\n",pivot[0],pivot[1],pivot[2]);
        for(int i=1;i<NumTasks;i++){
            MPI_Send(&pivot,3,MPI_INT,i,55,MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(&pivot,3,MPI_INT,0,55,MPI_COMM_WORLD,&mpistat);
        //printf("TID%i: received data=%d %d %d\n", SelfTID,pivot[0],pivot[1],pivot[2]);
    }


    /*function find_distances()*/
    double sumOfSquares=0;
    int a=0;
    int start=SelfTID*sub_n;
    int end=SelfTID*sub_n + sub_n;
    for (int i=start;i<end;i++){
        sumOfSquares=0;
        for(int j=0;j<d;j++){
            a=pow(array[i][j]-pivot[j],2);
            sumOfSquares+=a;
        }
        if(SelfTID==0){
            all_distances[i-start]=sqrt(sumOfSquares);
            distances[i-start]=sqrt(sumOfSquares);
        }
        else{
            distances[i-start]=sqrt(sumOfSquares);
        }
    }

    /*for (int i =0; i<n/NumTasks;i++){
        printf("%i %f\n ",SelfTID, distances[i]);
    }*/
    //end_function

    /*send to process 0 the calculated distances*/
    if(SelfTID != 0){
        MPI_Send(&distances,sub_n,MPI_DOUBLE,0,55,MPI_COMM_WORLD);
    }
    else{
        for(int i=1;i<NumTasks;i++){
            MPI_Recv(&(all_distances[i*sub_n]),sub_n,MPI_DOUBLE,i,55,MPI_COMM_WORLD,&mpistat);
            //printf("TID%i: received data=%d %d %d\n", SelfTID,pivot[0],pivot[1],pivot[2]);
        }
    }
    /*if(SelfTID==0){
        for (int i =0; i<n;i++){
            printf(" %f\n ", all_distances[i]);
        }
    }*/

    /*Announce median_distance*/
    if(SelfTID == 0){
        median_distance= quickselect(all_distances, 0, n-1, n/2);
        printf("median_distance%f\n ",median_distance);
        for(int i=1;i<NumTasks;i++){
            MPI_Send(&median_distance,1,MPI_DOUBLE,i,55,MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(&median_distance,1,MPI_DOUBLE,0,55,MPI_COMM_WORLD,&mpistat);
        printf("TID%i: received median value=%f\n", SelfTID,median_distance);
    }

    /*function distribute_and_split()*/
    double array_smalls[sub_n];
    double array_bigs[sub_n];
    int smalls_pointer=0;
    int bigs_pointer=0;
    for (int i=0;i<sub_n;i++){
        if(distances[i]<median_distance){
            array_smalls[smalls_pointer]=distances[i];
            smalls_pointer++;
        }
        else{
            array_bigs[bigs_pointer]=distances[i];
            bigs_pointer++;
        }
    }
    printf("size_1 %d\n ",smalls_pointer);
    printf("size_2 %d\n ",bigs_pointer);
    //end_function

    /*Exchange values*/
    int data_size=0;
    if(SelfTID == 0){
        MPI_Send(&bigs_pointer,1,MPI_DOUBLE,1,55,MPI_COMM_WORLD);
    }
    else{
        MPI_Recv(&data_size,1,MPI_DOUBLE,0,55,MPI_COMM_WORLD,&mpistat);
        printf("TID%i: received data_size=%d\n", SelfTID,data_size);
    }
    if(SelfTID == 0){
        MPI_Send(&array_bigs,bigs_pointer,MPI_DOUBLE,1,55,MPI_COMM_WORLD);
    }
    else{
        MPI_Recv(&array_bigs[bigs_pointer],data_size,MPI_DOUBLE,0,55,MPI_COMM_WORLD,&mpistat);
    }
    if(SelfTID == 1){
        MPI_Send(&smalls_pointer,1,MPI_DOUBLE,0,55,MPI_COMM_WORLD);
    }
    else{
        MPI_Recv(&data_size,1,MPI_DOUBLE,1,55,MPI_COMM_WORLD,&mpistat);
        printf("TID%i: received data_size=%d\n", SelfTID,data_size);
    }
    if(SelfTID == 1){
        MPI_Send(&array_smalls,smalls_pointer,MPI_DOUBLE,0,55,MPI_COMM_WORLD);
    }
    else{
        MPI_Recv(&array_smalls[smalls_pointer],data_size,MPI_DOUBLE,1,55,MPI_COMM_WORLD,&mpistat);
    }

    if(SelfTID== 0){
      for (int i =0; i<sub_n;i++){
          printf("%i  %f\n ", SelfTID,array_smalls[i]);
      }
    }else{
      for (int i =0; i<sub_n;i++){
          printf(" %i %f\n ", SelfTID, array_bigs[i]);
      }
    }

    MPI_Finalize();
    return 0;
}

// Returns the k'th smallest element in the list within `left…right`
// (i.e., left <= k <= right). The search space within the array is
// changing for each round – but the list is still the same size.
// Thus, `k` does not need to be updated with each round.
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
