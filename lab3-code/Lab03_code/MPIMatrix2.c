#include<stdio.h>
#include<mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <cstdio>
#include <cmath>

#define EPS 1e-2
//matrix multiplication
void MatrixMul(float *A, float *B, float *C, int m, int k,int n){
		int i,j,p;
		float sum;
		for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (p=0; p < k; p++) {
                sum += A[i*k+p]*B[p*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

//check the correctness
int check(float *P_serial,float *P_para,int m){
  for(int i = 0; i < m; ++i)
    if(fabs(P_serial[i]-P_para[i])>EPS){
      printf("error!\n");
      return 0;
    }
  printf("same!\n");
  return 1;

}

// print matrix
void Print(float* M, int row, int col)
{
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            printf("%.2f\t", M[i*col+j]);
        }
        printf("\n");
    }  
}

int main (int argc, char *argv[])
{
    int rank;
    int number_of_processes;
    int m,n,k,srow,saveM;
    float *M, *N, *P_para,*P_serial,*proc_M,*proc_P;
    double start_para,end_para,start_serial,end_serial;
		
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
	
    // Read Size of two Matrices
    if (rank == 0)
    {
		  	printf("Please input the size of two matrices(m*k,k*n):\n");
		  	fflush(stdout);
				scanf("%d %d %d", &m,&k,&n);
				saveM=m;
				if(m%number_of_processes!=0){
					m-=m%number_of_processes;
					m+=number_of_processes;
				}
		}
		MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);		
    
		size_t sizeM = m * k * sizeof(float);
		size_t sizeN = n * k * sizeof(float);
		size_t sizeP = m * n * sizeof(float);
    M = (float *) malloc(sizeM);
		N = (float *) malloc(sizeN);
		P_serial = (float *) malloc(sizeP);
	  P_para = (float *)malloc(sizeP);
		
    /* broadcast matrix N */
    srow = m / number_of_processes;

    printf("process %d calculates %d rows\n", rank, srow);
    proc_M=(float*)malloc(srow * k * sizeof(float));
    proc_P = (float *)malloc(srow * n * sizeof(float));
        
    if (rank == 0)
    {
        /* master code */
        /* allocate memory for matrix M, vectors y, and initialize them */
        // A = (double *)malloc(m * n * sizeof(double));
        //initialize
        srand(time(NULL));
        for(int i = 0; i < m; ++i)
        {
          for(int j=0;j<k;j++){
            if(i<saveM)
              M[i*k+j] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
            else{
              M[i*k+j] =0;
            }
            
          }	
        }
        for(int i = 0; i < k*n; ++i)
        {
            N[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
        }
        
        start_para = MPI_Wtime();
        /* send sub-matrices to other processes */
        
        MPI_Scatter(M, srow*k, MPI_FLOAT, proc_M, srow*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        MPI_Bcast(N, n*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //calculate local result
        MatrixMul(proc_M, N, proc_P, srow, k, n);
        
        //gather results
        MPI_Gather(proc_P, srow*n, MPI_FLOAT, P_para, srow*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        
        end_para = MPI_Wtime();
        start_serial = MPI_Wtime();
        MatrixMul(M,N,P_serial,saveM,k,n);
        end_serial = MPI_Wtime();
        
        //print result
        printf("The size of the two matrices: %d*%d,%d*%d\n",saveM,k,k,n);
        printf("CPU runs %3f seconds\n", end_serial-start_serial);
        
        printf("MPI runs %3f seconds\n", end_para-start_para);
        //Compare the results
        check(P_serial,P_para,saveM*n);
        float ratio=(end_serial-start_serial)/(end_para-start_para);
        printf("ratio=%f\n",ratio);

    }
    else
    {
        
        printf("This is rank %d running.\n", rank);
        MPI_Scatter(M, srow*k, MPI_FLOAT, proc_M, srow*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        MPI_Bcast(N, n*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //calculate local result
        MatrixMul(proc_M, N, proc_P, srow, k, n);
        
        //gather results
        MPI_Gather(proc_P, srow*n, MPI_FLOAT, P_para, srow*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        printf("This is rank %d ending.\n", rank);
    }
      
		free(M);
    free(N);
    free(P_serial);
    free(P_para);
    free(proc_M);
    free(proc_P);
    
    MPI_Finalize();
    return 0;
}
