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
    int send_id = rank;    // 当前进程发送的数据块的标识，向哪个进程发送数据
    int recv_id = (rank - 1 + number_of_processes) % number_of_processes;     // 当前进程接收的数据块的标识，收到来自哪个进程的数据
    float *buff = (float *)malloc(sizeof(float) * batch_size);
    int *counts = (int *)malloc(sizeof(int) * number_of_processes);      // 每个进程的元素数量
    int *displs = (int *)malloc(sizeof(int) * number_of_processes);      // 每个进程的起始索引
    MPI_Status status;
    MPI_Request request;

    for (int i = 0; i < number_of_processes; i ++)
    {
        int start_index = i * batch_size;
        if (start_index >= n)
        {
            counts[i] = 0;
        }
        else if (start_index + batch_size <= n)
        {
            counts[i] = batch_size;
        }
        else
        {
            counts[i] = n - start_index;
        }
        displs[i] = start_index;
    }

    for (int i = 0; i < number_of_processes - 1; i ++)
    {
        int send_count = counts[send_id];
        int recv_count = counts[recv_id];
        // MPI_Isend(res + batch_size * send_id, batch_size, datatype, (rank + 1) % number_of_processes, 0, comm, &request);
        MPI_Isend(res + displs[send_id], send_count, datatype, (rank + 1) % number_of_processes, 0, comm, &request);
        // MPI_Recv(buff, batch_size, datatype, (rank - 1 + number_of_processes) % number_of_processes, 0, comm, &status);
        MPI_Recv(buff, recv_count, datatype, (rank - 1 + number_of_processes) % number_of_processes, 0, comm, &status);
        MPI_Wait(&request, &status);

        // for (int j = 0; j < batch_size; j ++)
        // {
        //     int res_id = recv_id * batch_size + j;
        //     if (op == MPI_SUM)
        //     {
        //         res[res_id] += buff[j]; 
        //     }
        //     else if (op == MPI_MAX)
        //     {
        //         res[res_id] = buff[j] > res[res_id] ? buff[j] : res[res_id];
        //     }
        // }
        for (int j = 0; j < recv_count; j++)
        {
            int res_id = displs[recv_id] + j;
            if (op == MPI_SUM)
            {
                res[res_id] += buff[j];
            }
            else if (op == MPI_MAX)
            {
                res[res_id] = (buff[j] > res[res_id]) ? buff[j] : res[res_id];
            }
        }

        send_id = recv_id;
        recv_id = (recv_id - 1 + number_of_processes) % number_of_processes;
    }
    for (int step = 0; step < number_of_processes - 1; step++){
        int send_count = counts[send_id];
        int recv_count = counts[recv_id];
        float *send_buff = res + send_id * batch_size;
        // use isend to avoid deadlock
        MPI_Isend(send_buff, send_count, datatype, (rank + 1) % number_of_processes, 0, comm, &request);
        MPI_Recv(buff, recv_count, datatype, (rank - 1 + number_of_processes) % number_of_processes, 0, comm, &status);
        MPI_Wait(&request, &status);

        for (int i = 0; i < recv_count; i++){
            int res_id = recv_id * batch_size + i;
            //int res_id = displs[recv_id] + i;
            res[res_id] = buff[i];
        }

        send_id = recv_id;
        recv_id = (recv_id - 1 + number_of_processes) % number_of_processes;
    }
    free(buff);
    free(counts);
    free(displs);
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
    // Print(array, n);
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
    // Print(res1, n);
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
    // Print(res2, n);
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