
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void find_moment ( int ** A, int ** B ,int L );

int main(int argc, char *argv[]){

    int ** I;
    int ** J;
    int L = 1000;
    int N= L*L;
    int k = 40;
    int spin;
    int r;

    struct timeval startwtime, endwtime;
    double seq_time;
    srand(time(NULL));

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

    gettimeofday (&startwtime, NULL);


    for (int z=0;z<k;z++){
        if(z%2==0){
            find_moment(I,J,L);
        }
        else{
            find_moment(J,I,L);
        }
    }

    gettimeofday (&endwtime, NULL);

    seq_time = (endwtime.tv_sec -startwtime.tv_sec)*1000000L  +endwtime.tv_usec - startwtime.tv_usec ;
    printf("time=%f\n",seq_time);

    /*printf("Final array\n");
    if(k%2==0){
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
