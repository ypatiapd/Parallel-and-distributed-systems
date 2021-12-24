#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

#define SWAP(x, y) { double temp = x; x = y; y = temp; }

struct timeval startwtime, endwtime;
double seq_time;

int findPreviousPowerOf2(int n);
double quickselect(double nums[], int left, int right, int k);
double partition(double a[], int left, int right, int pIndex);
double distributeByMedian(int n, int d,int all_tasks,int NumTasks,int depth,int leader,int mean_id, double Pivot[d],int SelfTID,MPI_Status mpistat);

double *array;

int main(int argc, char *argv[])
{
    int n; //number of points
    int d; //number of attributes
    int trash=0;
    FILE *ptr;
    int z=0;
    int SelfTID,NumTasks,t,data;
    MPI_Status mpistat;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&SelfTID);
    int depth=log2(NumTasks);// the depth we want to reach , in every regression call we substract 1 ,until it reaches 0
    int leader=0; // leader of first regression call
    int all_tasks=NumTasks; //number of processes
    int mean_id=NumTasks/2;

    ptr = fopen(argv[1],"rb");  // r for read, b for binary
    if (ptr == NULL){
        printf("no file \n ");
        return -1;
    }
    z=fread(&d,sizeof(d),1,ptr); // read number of attributes
    z=fread(&trash,sizeof(trash),1,ptr);
    z=fread(&n,sizeof(n),1,ptr); // read number of points
    z=fread(&trash,sizeof(trash),1,ptr);
    //printf("n= %d\n ", n);
    printf("d= %d\n ", d);
    n= findPreviousPowerOf2(n);  //convert n to previous power of 2 for easier implementation
    //int next = pow(2, ceil(log(n)/log(2)));   if we want next power of 2
    printf("n= %d\n ", n);

    double pivot[d];
    int block=n/NumTasks;

    array = (double *) malloc(block*d * sizeof(double));

    double buffer[d];
    int start=SelfTID*block;  //file reading indexes for each process
    int end =start+block;
    gettimeofday (&startwtime, NULL);
    for (int i = 0; i < n; i++){
       for (int j = 0; j < d; j++){
            fread(&buffer[j], sizeof(double), 1, ptr);//read from file
         // Be sure to check status of fread
       }
       if((i>=start)&&(i<end)){ //each process keeps only the block that is assigned to it
            for (int j=0;j<d;j++)
                array[(i-start)*d+j]=buffer[j];
       }
    }
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
  		      + endwtime.tv_sec - startwtime.tv_sec);
    printf("TID=%i  file reading time=%f\n",SelfTID,seq_time);

    fclose(ptr);

    /*if(SelfTID==3){
        for (int i =0; i<block*d;i++){
                printf("bin= %lf \n ", array[i]);
            printf("\n");
        }
    }*/
    gettimeofday (&startwtime, NULL);
    distributeByMedian(n,d,all_tasks,NumTasks,depth,leader,mean_id,pivot,SelfTID,mpistat);
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
  		      + endwtime.tv_sec - startwtime.tv_sec);
    printf("TID=%i exec time=%f\n",SelfTID,seq_time);
    MPI_Finalize();
    return 0;
}

int findPreviousPowerOf2(int  n)
{
    // do till only one bit is left
    while (n & n - 1) {
        n = n & n - 1;        // unset rightmost bit
    }
    // `n` is now a power of two (less than or equal to `n`)
    return n;
}

double distributeByMedian(int n, int d,int all_tasks,int NumTasks,int depth,int leader,int mean_id, double Pivot[d],int SelfTID,MPI_Status mpistat){
    int block=n/NumTasks;
    double pivot[d];
    double *distances;// all distances of each process
    double *all_distances;// all distances of all processes, used only by the leaders

    all_distances = (double *) malloc(n * sizeof(double));
    distances = (double *) malloc(block * sizeof(double));
    double median_distance;
    int choose=0; //pivot index choose
    time_t t;
    srand((unsigned) time(&t));
    //printf("depth %d hello from %i of %i tasks\n",depth ,SelfTID,NumTasks);

    /*choose and announce pivot ,receive pivot from the other processes*/
    if(depth==log2(all_tasks)){
        if(SelfTID == leader){
            choose=rand() % block;
            printf("depth %d TID %i index choose %d \n",depth,SelfTID,choose);
            for(int i=0;i<d;i++){
                pivot[i]=array[choose*d+i];
            }
            //printf("depth %d TID %i ,chose pivot\n",depth,SelfTID);
            for(int i=leader+1;i<leader+NumTasks;i++){
                MPI_Send(&pivot,d,MPI_DOUBLE,i,55,MPI_COMM_WORLD);
            }
        }
        else{
            MPI_Recv(&pivot,d,MPI_DOUBLE,leader,55,MPI_COMM_WORLD,&mpistat);
            printf("TID%i: received pivot\n", SelfTID);
        }
    }
    else{
        for(int i=0;i<d;i++){
            pivot[i]=Pivot[i];
        }
    }
    /*end*/

    /*function find_distances()*/
    double sumOfSquares=0;
    double a=0;

    for (int i=0;i<block*d;i+=d){
        sumOfSquares=0;
        for(int j=0;j<d;j++){
            a=pow(array[i+j]-pivot[j],2);
            sumOfSquares+=a;
        }
        if(SelfTID==leader){
            all_distances[i/d]=sumOfSquares;
            distances[i/d]=sumOfSquares;
            //printf("TID=%i  calculated distance  %f pointer %d \n ",SelfTID,sqrt(sumOfSquares),i);
        }
        else{
            distances[i/d]=sumOfSquares;
            //printf("TID=%i  calculated distance  %f pointer %d \n ",SelfTID,sqrt(sumOfSquares),i);
        }
    }
    /*end*/

    /*if(depth==2){
        for (int i =0; i<block;i++){
            printf("TID:%i  %f\n ", SelfTID,distances[i]);
        }
    }*/

    /*if((SelfTID==0)&&(depth==2)){
        for (int i =0; i<n;i++){
            printf(" %f\n ", all_distances[i]);
        }
    }*/
    //end_function

    /*send to leader the calculated distances*/
    if(SelfTID != leader){
        MPI_Send(distances,block,MPI_DOUBLE,leader,55,MPI_COMM_WORLD);
    }
    else{
        for(int i=leader+1;i<leader+NumTasks;i++){
            MPI_Recv(&(all_distances[(i-leader)*block]),block,MPI_DOUBLE,i,55,MPI_COMM_WORLD,&mpistat);
            //printf("TID%i: received data=%d %d %d\n", SelfTID,pivot[0],pivot[1],pivot[2]);
        }
    }
    /*end*/

    /*if((SelfTID==0)&&(depth==2)){
        for (int i =0; i<n;i++){
            printf(" %f\n ", all_distances[i]);
        }
    }*/

    /*Announce median_distance*/
    if(SelfTID == leader){
        median_distance= quickselect(all_distances, 0, n-1, n/2);
        printf("depth = %d TID leader = %i median_distance%f\n ",depth,SelfTID,median_distance);
        for(int i=leader+1;i<leader+NumTasks;i++){
            MPI_Send(&median_distance,1,MPI_DOUBLE,i,55,MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(&median_distance,1,MPI_DOUBLE,leader,55,MPI_COMM_WORLD,&mpistat);
        printf("depth= %d TID%i: received median value=%f\n",depth, SelfTID,median_distance);
    }
    /*end*/

    /*function distribute_and_split()*/
    int *array_smalls; //array with indexes of the points with distance smaller than the median
    int *array_bigs;  //array with indexes of the points with distance bigger than the median
    array_bigs = (int *) malloc(block * sizeof(int));
    array_smalls = (int *) malloc(block * sizeof(int));
    int smalls_counter=0; //counts the small distance points
    int bigs_counter=0;  //counts the big distance points

    for (int i=0;i<block;i++){
        if(distances[i]<median_distance){
            //printf(" small distance  %f pointer %d \n ",distances[i],start+i);
            array_smalls[smalls_counter]=i*d;
            smalls_counter++;
        }
        else if(distances[i]>=median_distance){
            //printf("tid %i big distance  %f \n ",SelfTID,distances[i]);
            //printf(" big distance  %f pointer %d \n ",distances[i],start+i);
            array_bigs[bigs_counter]=i*d;
            bigs_counter++;
        }
    }
    printf(" depth %d tid %i smalls %d \n ",depth,SelfTID,smalls_counter);
    printf(" depth %d tid %i bigs %d \n ", depth,SelfTID, bigs_counter);

    free(all_distances);
    free(distances);
    /*end*/

    /*sizes exchange*/
    int sizes[NumTasks];//array with number of small distance or big distance points in each process, depending on the id of the process
    if(SelfTID<mean_id)
        sizes[SelfTID-leader]=smalls_counter;
    else
        sizes[SelfTID-leader]=bigs_counter;

    for (int i=leader;i<leader+NumTasks/2;i++){
        if(SelfTID!=i){
            MPI_Send(&smalls_counter,1,MPI_INT,i,55,MPI_COMM_WORLD);
        }
    }
    if(SelfTID<mean_id){
        for(int i=leader;i<leader+NumTasks;i++){
            if(SelfTID!=i){
              MPI_Recv(&sizes[i-leader],1,MPI_INT,i,55,MPI_COMM_WORLD,&mpistat);
            }
        }
    }
    for (int i=mean_id;i<mean_id+NumTasks/2;i++){
        if(SelfTID!=i){
            MPI_Send(&bigs_counter,1,MPI_INT,i,55,MPI_COMM_WORLD);
        }
    }
    if(SelfTID>=mean_id){
        for(int i=leader;i<leader+NumTasks;i++){
            if(SelfTID!=i){
              MPI_Recv(&sizes[i-leader],1,MPI_INT,i,55,MPI_COMM_WORLD,&mpistat);
            }
        }
    }
    for (int i =0; i<NumTasks;i++){
        printf("depth= %d TID= %i sizes %d\n ", depth, SelfTID,sizes[i]);
    }
    /*end*/

    /*points exchange*/
    double *buffer;
    double *recv_buffer;
    int pfs=0;  // prefix scan pointer of received points
    int my_pfs=0;// the point that each process will start saving the data from the received points
    MPI_Request mpireq;

    if(SelfTID<mean_id){
        int size=0; //counter that counts the saved points from each process
        for(int i=0;i<NumTasks/2;i++){ //find pfs of each process
            if(leader+i==SelfTID)
                break;
            my_pfs+=(block-sizes[i])*d;
        }
        printf("TID %i  my_pfs %d \n ",SelfTID,my_pfs);
        buffer = (double *) malloc(bigs_counter*d * sizeof(double));
        for(int i=0;i<bigs_counter*d;i+=d){ //transfer to buffer the data that will be send
            for(int j=0;j<d;j++){
                buffer[i+j]=array[array_bigs[i/d]+j];
            }
        }
        for(int i=0;i<NumTasks/2;i++){//send the big distance points to the other group(id>mean_id) processes
            MPI_Isend(buffer,bigs_counter*d,MPI_DOUBLE,mean_id+i,55,MPI_COMM_WORLD,&mpireq );
        }
        for (int i =0; i<smalls_counter*d;i+=d){ //restore at the start of the array tha small distance points
            for(int j=0;j<d;j++)
                array[i+j]=array[array_smalls[i/d]+j];
        }
        int rest=0;//counter for the total points that a process need to complete
        int pointer=smalls_counter*d; //pointer to the array of the process where the points are stored.
        for(int i=0;i<NumTasks/2;i++){
            size=0;
            recv_buffer = (double *) malloc(sizes[NumTasks/2+i]*d * sizeof(double));
            MPI_Recv(recv_buffer,sizes[NumTasks/2+i]*d,MPI_DOUBLE,mean_id+i,55,MPI_COMM_WORLD,&mpistat);
            //for (int j=0;j<10 ;j++)
            //    printf("TID %i buff %f \n ",SelfTID,recv_buffer[j]);
            int buffer_idx=my_pfs-pfs+rest;
            pfs+=sizes[NumTasks/2+i]*d;
            printf("TID %i first bufer_idx %d \n ",SelfTID,buffer_idx);
            printf("TID %i pfs %d \n ",SelfTID,pfs);
            //&&(pfs<my_pfs+bigs_pointer*d)
            printf("TID %i pointer %d start iter %d \n ",SelfTID,pointer,i);
            if ((pfs>my_pfs)&&(rest<bigs_counter*d)&&(buffer_idx>=0)){
                printf("TID %i pfs %d my_pfs %d \n ",SelfTID,pfs,my_pfs);
                while((rest<bigs_counter*d)&&(buffer_idx+size<sizes[NumTasks/2+i]*d)){
                    array[pointer+size]=recv_buffer[buffer_idx+size];
                    size+=1;
                    rest+=1;
                    //printf("TID %i bufer %f \n ",SelfTID,recv_buffer[buffer_idx+size]);
                    //pointer-smalls_counter*d+size
                    //printf("TID %i pointer1 %d \n ",SelfTID,pointer-smalls_counter*d+size);
                    //printf("TID %i pointer2 %d \n ",SelfTID,bigs_counter*d);
                }
                pointer+=size;
                printf("TID %i pointer %d end iter %d \n ",SelfTID,pointer,i);
            }
            free(recv_buffer);
        }
        //free(buffer);
    }
    else{//same procedure fore the points with id>mean_id
        int size=0;
        for(int i=0;i<NumTasks;i++){
            if(mean_id+i==SelfTID)
                break;
            my_pfs+=(block-sizes[i+NumTasks/2])*d;
        }
        printf("TID %i  my_pfs %d \n ",SelfTID,my_pfs);
        buffer = (double *) malloc(smalls_counter*d * sizeof(double));
        for(int i=0;i<smalls_counter*d;i+=d){
            for(int j=0;j<d;j++){
                buffer[i+j]=array[array_smalls[i/d]+j];
            }
        }
        for(int i=0;i<NumTasks/2;i++){
            MPI_Isend(buffer,smalls_counter*d,MPI_DOUBLE,leader+i,55,MPI_COMM_WORLD,&mpireq );
            //sleep(3);
        }
        for (int i =0; i<bigs_counter*d;i+=d){
            for(int j=0;j<d;j++)
                array[i+j]=array[array_bigs[i/d]+j];
        }
        int pointer=bigs_counter*d;
        int rest=0;
        for(int i=0;i<NumTasks/2;i++){
            size=0;
            recv_buffer = (double *) malloc(sizes[i]*d * sizeof(double));
            MPI_Recv(recv_buffer,sizes[i]*d,MPI_DOUBLE,leader+i,55,MPI_COMM_WORLD,&mpistat);
            //for (int j=0;j<10 ;j++)
            //    printf("TID %i buff %f \n ",SelfTID,recv_buffer[j]);
            int buffer_idx=my_pfs-pfs+rest;//edwwwwwww gamwwwwww
            pfs+=sizes[i]*d;
            printf("TID %i first bufer_idx %d \n ",SelfTID,buffer_idx);
            printf("TID %i pfs %d \n ",SelfTID,pfs);
            printf("TID %i pointer %d start iter %d \n ",SelfTID,pointer,i);
            if ((pfs>my_pfs)&&(rest<smalls_counter*d)&&(buffer_idx>=0)){
                printf("TID %i pfs %d my_pfs %d \n ",SelfTID,pfs,my_pfs);
                //printf("TID %i irthaaaaaa \n ",SelfTID);
                while((rest<smalls_counter*d)&&(buffer_idx+size<sizes[i]*d)){
                    array[pointer+size]=recv_buffer[buffer_idx+size];
                    size+=1;
                    rest+=1;
                    //printf("TID %i bufer %f \n ",SelfTID,recv_buffer[buffer_idx+size]);
                    //pointer-smalls_counter*d+size
                    //printf("TID %i pointer1 %d \n ",SelfTID,pointer-smalls_counter*d+size);
                    //printf("TID %i pointer2 %d \n ",SelfTID,bigs_counter*d);
                }
                pointer+=size;
                printf("TID %i pointer %d end iter %d \n ",SelfTID,pointer,i);
            }
            free(recv_buffer);
        }
        //free(buffer);
    }
    /*if(SelfTID==3){
        for (int j=0;j<block*d;j++)
            printf("buff %f \n ",array[j]);
    }*/

    /*end*/
    free(array_bigs);
    free(array_smalls);

    /*validation*/
    sumOfSquares=0;
    for (int i=0;i<block*d;i+=d){
        sumOfSquares=0;
        for(int j=0;j<d;j++){
            a=pow(array[i+j]-pivot[j],2);
            sumOfSquares+=a;
        }
        //if(depth==1)
            //printf("TID %i     %lf\n ",SelfTID, sumOfSquares);
       if(SelfTID<mean_id){
            if(sumOfSquares>median_distance)
                  printf("iter %d depth %d TID %i small distance   %f > %f  \n",i, depth,SelfTID,sumOfSquares,median_distance );
        }
        else{
            if(sumOfSquares<median_distance)
                  printf("iter %d depth %d TID %i big distance %f < %f\n",i, depth ,SelfTID,sumOfSquares,median_distance);
        }
    }
    /*end*/

    depth--;
    int leader1=leader;
    int leader2=leader+NumTasks/2;
    int mean_id1=leader+NumTasks/4;
    int mean_id2=leader2+NumTasks/4;

    /*regression*/

    if (depth>0){
        //printf("TID %i  bika  %d \n ",SelfTID,depth);
        printf(" depth=%d NumTasks %d  leader1 %d leader2 %d mean1 %d mean2 %d \n ",depth,NumTasks,leader1,leader2,mean_id1,mean_id2);

        if(SelfTID<mean_id){
            distributeByMedian(n/2,d,all_tasks,NumTasks/2,depth,leader1,mean_id1,pivot,SelfTID,mpistat);
        }
        if(SelfTID>=mean_id){
            distributeByMedian(n/2,d,all_tasks,NumTasks/2,depth,leader2,mean_id2,pivot,SelfTID,mpistat);
        }
    }
    /*end*/

    //printf("TID %i  vgika %d \n ",SelfTID,depth);
    return 0;
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
