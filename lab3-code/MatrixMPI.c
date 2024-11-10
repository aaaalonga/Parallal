#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cstdio>
#include <math.h>
#define EPS 1e-2
// Init
void Init(float *M, float *N, int m, int k, int n)
{
    // srand(time(NULL));
    for (int i = 0; i < m * k; i ++)
        M[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
    for (int i = 0; i < k * n; i ++)
        N[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
}
// Print
void Print(float *M, float *N, int m, int k, int n)
{
    printf("Matrix M:\n");
    for (int i = 0; i < m; i ++)
    {
        for (int j = 0; j < k; j ++)
        {
            printf("%f ", M[i * k + j]);
        }
        printf("\n");
    }
    printf("Matrix N:\n");
    for (int i = 0; i < k; i ++)
    {
        for (int j = 0; j < n; j ++)
        {
            printf("%f ", N[i * n + j]);
        }
        printf("\n");
    }
}
// Matrix multiplication
void MatrixMul(float *M, float *N, float *P, int m, int k, int n){
    float sum;
    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < n; j ++) {
            sum = 0.0;
            for (int p = 0; p < k; p ++) {
                sum += M[i * k + p] * N[p * n + j];
            }
            P[i * n + j] = sum;
        }   
    }
}

// Parallal_Scatter
void Parallal_Scatter(float *M, float *N, float *M_scatter, float *P_scatter, float *P_t, int m, int k, int n, int number_of_processes)
{
    int each_row = ceil(m * 1.0 / number_of_processes);
    int m_new = each_row * number_of_processes;
    float *M_new = (float *)malloc(m_new * k * sizeof(float));
    float *P_new = (float *)malloc(m_new * n * sizeof(float));
    for (int i = 0; i < m_new; i ++)
    {
        for (int j = 0; j < k; j ++)
        {
            if (i < m)
            {
                M_new[i * k + j] = M[i * k + j];
            }
            else
            {
                M_new[i * k + j] = 0;
            }
        }
    }
    MPI_Scatter(M_new, each_row * k, MPI_FLOAT, M_scatter, each_row * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(N, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MatrixMul(M_scatter, N, P_t, each_row, k, n);
    MPI_Gather(P_t, each_row * n, MPI_FLOAT, P_new, each_row * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < m; i ++)
    {
        for (int j = 0; j < n; j ++)
        {
            P_scatter[i * n + j] = P_new[i * n + j];
        }
    }
    
    free(M_new);
    free(P_new);
}

// Parallal_Send
void Parallal_Send(float *M, float *N, float *P_send, int m, int k, int n, int each_row, int number_of_processes)
{
    for(int i = 1; i < number_of_processes; i ++)
    {
        MPI_Send(M + (i - 1) * k * each_row, each_row * k, MPI_FLOAT, i, 6, MPI_COMM_WORLD);
    }
    MPI_Bcast(N, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = (number_of_processes - 1) * each_row; i < m; i ++)
    {
        for (int j = 0; j < n; j ++)
        {
            float temp = 0.0;
            for (int p = 0; p < k; p ++)
            {
                temp += M[i * k + p] * N[p * n + j];
            }
            P_send[i * n + j] = temp; 
        }
    }
    for (int i = 1; i < number_of_processes; i ++)
    {
        MPI_Recv(P_send + (i - 1) * each_row * n, each_row * n, MPI_FLOAT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
}
// Parallal_Recv
void Parallal_Recv(float *M_send, float *N, float *P_temp, int each_row, int k, int n)
{
    MPI_Recv(M_send, each_row * k, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Bcast(N, n * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Print(M_send, N, each_row, k, n);
    MatrixMul(M_send, N, P_temp, each_row, k, n);
    MPI_Send(P_temp, each_row * n, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    int m, k, n;
    FILE *file1 = fopen("input1.txt","r");
    FILE *file2 = fopen("output1.txt", "w");
    fscanf(file1, "%d,%d,%d", &m, &k, &n);
    m = 100, k = 500, n = 100;
    float *M, *N, *P_send, *S, *P_scatter;
    float *M_send, *P_temp;
    float *M_scatter, *P_t;
    int rank, number_of_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int each_row = m / number_of_processes;

    size_t sizeN = k * n * sizeof(float);
    size_t sizeM = m * k * sizeof(float); 
    size_t sizeP = m * n * sizeof(float);

    M_send = (float *)malloc(each_row * k * sizeof(float));
    M_scatter = (float *)malloc(ceil(m * 1.0 / number_of_processes) * k * sizeof(float));
    N = (float *)malloc(sizeN);
    P_temp = (float *)malloc(each_row * n * sizeof(float));
    P_t = (float *)malloc(ceil(m * 1.0 / number_of_processes) * n * sizeof(float));
    M = (float *)malloc(sizeM);
    P_scatter = (float *)malloc(sizeP);
    

    if (rank == 0)
    {
        P_send = (float *)malloc(sizeP);
        S = (float *)malloc(sizeP);

        Init(M, N, m, k, n);
        // Print(M, N, m, k, n);

        //Serial
        double start_serial = MPI_Wtime();
        MatrixMul(M, N, S, m, k, n);
        double end_serial = MPI_Wtime();
        printf("CPU runs %.3f ms\n", (end_serial - start_serial) * 1000);
        fprintf(file2, "%.3f,", (end_serial - start_serial) * 1000);

        //Parallal_Recv
        double start_parallal_recv = MPI_Wtime();
        Parallal_Send(M, N, P_send, m, k, n, each_row, number_of_processes);
        double end_parallal_recv = MPI_Wtime();
        printf("MPI_RECV runs %.3f ms\n", (end_parallal_recv - start_parallal_recv) * 1000);

        //Check
        int flag1 = 0;
        for (int i = 0; i < m * n; i ++){
            if (fabs(S[i] - P_send[i]) > EPS)
            {
                flag1 = 1;
                printf("Serial result and MPI_RECV result differ\n");
                printf("%d %f %f\n", i, S[i], P_send[i]);
            }
        }
        if (flag1 == 0)
                printf("Serial result and MPI_RECV result same\n");

        free(P_send);
    }
    else
    {
        Parallal_Recv(M_send, N, P_temp, each_row, k, n);
    }

    //Parallal_Gather
    double start_parallal_gather = MPI_Wtime();
    Parallal_Scatter(M, N, M_scatter, P_scatter, P_t, m, k, n, number_of_processes);
    
    if(rank == 0)
    {
        double end_parallal_gather = MPI_Wtime();
        printf("MPI_Gather runs %.3f ms\n", (end_parallal_gather - start_parallal_gather) * 1000);
        fprintf(file2, "%.3f", (end_parallal_gather - start_parallal_gather) * 1000);
        int flag2 = 0;
        for (int i = 0; i < m * n; i ++){
            if (fabs(S[i] - P_scatter[i]) > EPS)
            {
                flag2 = 1;
                printf("Serial result and MPI_Gather result differ\n");
                printf("%d %f %f\n", i, S[i], P_scatter[i]);
            }
        }
        if (flag2 == 0)
            printf("Serial result and MPI_Gather result same\n");
    
        free(S);
    }

    free(N);
    free(M);
    free(M_send);
    free(M_scatter);
    free(P_temp);
    free(P_t);
    free(P_scatter);
    fclose(file1);
    fclose(file2);
    MPI_Finalize();
    return 0;
}