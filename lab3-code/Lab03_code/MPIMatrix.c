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
int main (int argc, char *argv[])
{
    int rank;
    int number_of_processes;
    int m,n,k,srow,srow_last;
    float *M, *N, *P_para,*P_serial;
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
				
				
				size_t sizeM = m * k * sizeof(float);
				size_t sizeN = n * k * sizeof(float);
				size_t sizeP = m * n * sizeof(float);
        M = (float *) malloc(sizeM);
				N = (float *) malloc(sizeN);
				P_serial = (float *) malloc(sizeP);
				
				//initialize
				srand(time(NULL));
				for(int i = 0; i < m * k; ++i)
				{
						M[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
					 
				}
				for(int i = 0; i < k*n; ++i)
				{
						N[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
				}
				
				start_serial = MPI_Wtime();
				MatrixMul(M,N,P_serial,m,k,n);
				end_serial = MPI_Wtime();
				
  			//print result
  			printf("The size of the two matrices: %d*%d,%d*%d\n",m,k,k,n);
        printf("CPU runs %3f seconds\n", end_serial-start_serial);
        
  			
    }
    start_para = MPI_Wtime();
    
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0)
          N = (float *)malloc(k * n * sizeof(float));
        /* broadcast matrix N */
        MPI_Bcast(N, n*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        srow = m / number_of_processes;
        srow_last = m - srow * (number_of_processes - 1);
        if (rank == number_of_processes - 1)
            srow = srow_last;
        printf("process %d calculates %d rows\n", rank, srow);
        if (rank == 0)
        {
            /* master code */
            /* allocate memory for matrix M, vectors y, and initialize them */
            // A = (double *)malloc(m * n * sizeof(double));
            P_para = (float *)malloc(m * n * sizeof(float));

            /* send sub-matrices to other processes */
            int i;
            for (i = 1; i < number_of_processes - 1; i++)
                MPI_Send(M + i * srow * k, srow * k, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
               
            MPI_Send(M + i * srow * k, srow_last * k, MPI_FLOAT, number_of_processes - 1, 0, MPI_COMM_WORLD);

            /* perform its own calculation for the 1st sub-matrix */
            MatrixMul(M, N, P_para, srow, k, n);

            /* collect results from other processes */
            for (i = 1; i < number_of_processes - 1; i++)
                MPI_Recv(P_para + i * srow*n, srow*n, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P_para + i * srow*n, srow_last*n, MPI_FLOAT, number_of_processes - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            end_para = MPI_Wtime();
            
            printf("MPI runs %3f seconds\n", end_para-start_para);
            //Compare the results
  					check(P_serial,P_para,m*n);
  					float ratio=(end_serial-start_serial)/(end_para-start_para);
 						printf("ratio=%f\n",ratio);
						
    				free(P_serial);
        }
        else
        {
            /* slave code */
            /* allocate memory for sub-matrix A, and sub-sector y */
            M = (float *)malloc(srow * k * sizeof(float));
            P_para = (float *)malloc(srow * n * sizeof(float));

            /* receive sub-matrix from process 0 */
            MPI_Recv(M, srow * k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* perform the calculation on the sub-matrix */
            MatrixMul(M, N, P_para, srow, k, n);

            /* send the results to process 0 */
            MPI_Send(P_para, srow*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
        
    
		
		
		
		free(M);
    free(N);
    free(P_para);
    
    MPI_Finalize();
    return 0;
}
