#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


int rank, number_of_processes;
// Init
void Init(float *array, int n)
{
    // srand(time(NULL));
    for (int i = 0; i < n; i ++)
        array[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
}

//Print
void Print(float *array, int n)
{
    for (int i = 0; i < n; i ++)
        printf("%f ", array[i]);
    printf("\n");
}

//RING_Allreduce
void RING_Allreduce(float *res, int n, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int batch_size = n / number_of_processes + (n % number_of_processes != 0 ? 1 : 0);
    int send_id = rank;
    int recv_id = (rank - 1 + number_of_processes) % number_of_processes;
    float *buff = (float *)malloc(sizeof(float) * batch_size);
    MPI_Status status;
    MPI_Request request;

    for (int i = 0; i < number_of_processes - 1; i ++)
    {
        MPI_Isend(res + batch_size * send_id, batch_size, datatype, (rank + 1) % number_of_processes, 0, comm, &request);
        MPI_Recv(buff, batch_size, datatype, (rank - 1 + number_of_processes) % number_of_processes, 0, comm, &status);
        MPI_Wait(&request, &status);
        for (int j = 0; j < batch_size; j ++)
        {
            int res_id = recv_id * batch_size + j;
            if (op == MPI_SUM)
            {
                res[res_id] += buff[j]; 
            }
            else if (op == MPI_MAX)
            {
                res[res_id] = buff[j] > res[res_id] ? buff[j] : res[res_id];
            }
        }
        send_id = recv_id;
        recv_id = (recv_id - 1 + number_of_processes) % number_of_processes;
    }
    for (int step = 0; step < number_of_processes - 1; step++){
        float *send_buff = res + send_id * batch_size;
        // use isend to avoid deadlock
        MPI_Isend(send_buff, batch_size, datatype, (rank + 1) % number_of_processes, 0, comm, &request);
        MPI_Recv(buff, batch_size, datatype, (rank - 1 + number_of_processes) % number_of_processes, 0, comm, &status);
        MPI_Wait(&request, &status);

        for (int i = 0; i < batch_size; i++){
            int res_id = recv_id * batch_size + i;
            res[res_id] = buff[i];
        }

        send_id = recv_id;
        recv_id = (recv_id - 1 + number_of_processes) % number_of_processes;
    }
    free(buff);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    FILE *file1 = fopen("input2.txt","r");
    FILE *file2 = fopen("output2.txt", "w");
    int n;
    char op[5];
    float *array, *res1, *res2;
    double start_time, end_time;
    fscanf(file1, "%d,%s", &n, op);

    array = (float *)malloc(n * sizeof(float));
    res1 = (float *)malloc(n * sizeof(float));
    res2 = (float *)malloc(n * sizeof(float));
    Init(array, n);
    Print(array, n);
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Allreduce
    start_time = MPI_Wtime();
    if (strcmp(op, "sum") == 0 || strcmp(op, "SUM") == 0)
    {
        MPI_Allreduce(array, res1, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    else if (strcmp(op, "max") == 0 || strcmp(op, "MAX") == 0)
    {
        MPI_Allreduce(array, res1, n, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    }
    else
    {
        printf("Please input sum(or SUM) or max(or MAX)\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    Print(res1, n);
    // printf("process %d MPI_Allreduce time: %.2lf ms\n", rank, (end_time - start_time * 1000));
    if (rank == 0)
    {
        printf("MPI_Allreduce time: %.2lf ms\n", (end_time - start_time) * 1000);
        fprintf(file2, "%.2lf,", (end_time - start_time) * 1000);
    }

    
    for (int i = 0; i < n; i++)
        res2[i] = array[i];

    // RING_Allreduce
    start_time = MPI_Wtime();
    if (strcmp(op, "sum") == 0 || strcmp(op, "SUM") == 0)
    {
        RING_Allreduce(res2, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    else if (strcmp(op, "max") == 0 || strcmp(op, "MAX") == 0)
    {
        RING_Allreduce(res2, n, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    }
    else
    {
        printf("Please input sum(or SUM) or max(or MAX)\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    Print(res2, n);
    if (rank == 0)
    {
        printf("RING_Allreduce time: %.2lf ms\n", (end_time - start_time) * 1000);
        fprintf(file2, "%.2lf", (end_time - start_time) * 1000);
    }

    free(array);
    free(res1);
    free(res2);
    MPI_Finalize();
    return 0;
}