# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <string.h>

int rank, num_of_processes;

void print(float *array, int length){
    for (int i = 0; i < length; i++){
        printf("%f ", array[i]);
    }
    printf("\n");
}

// ring-based allreduce
void RingAllreduce(float* res_array, int length, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    int batch_size = (int)(length / num_of_processes), send_id = rank, recv_id = (rank - 1 + num_of_processes) % num_of_processes;

    float *recv_buff = (float*)malloc(batch_size * sizeof(float));
    MPI_Status status;
    MPI_Request request;

    // sum or max
    for (int step = 0; step < num_of_processes - 1; step++){
        float *send_buff = res_array + send_id * batch_size;
        // use isend to avoid deadlock
        MPI_Isend(send_buff, batch_size, datatype, (rank + 1) % num_of_processes, 0, comm, &request);
        MPI_Recv(recv_buff, batch_size, datatype, (rank - 1 + num_of_processes) % num_of_processes, 0, comm, &status);
        MPI_Wait(&request, &status);

        for (int i = 0; i < batch_size; i++){
            int res_id = recv_id * batch_size + i;
            if (op == MPI_SUM)
                res_array[res_id] += recv_buff[i];
            else if (op == MPI_MAX)
                res_array[res_id] = fmax(recv_buff[i], res_array[res_id]);
        }

        send_id = recv_id;
        recv_id = (recv_id - 1 + num_of_processes) % num_of_processes;
    }


    // =
    for (int step = 0; step < num_of_processes - 1; step++){
        float *send_buff = res_array + send_id * batch_size;
        // use isend to avoid deadlock
        MPI_Isend(send_buff, batch_size, datatype, (rank + 1) % num_of_processes, 0, comm, &request);
        MPI_Recv(recv_buff, batch_size, datatype, (rank - 1 + num_of_processes) % num_of_processes, 0, comm, &status);
        MPI_Wait(&request, &status);

        for (int i = 0; i < batch_size; i++){
            int res_id = recv_id * batch_size + i;
            res_array[res_id] = recv_buff[i];
        }

        send_id = recv_id;
        recv_id = (recv_id - 1 + num_of_processes) % num_of_processes;
    }

    free(recv_buff);
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);

    srand(time(0) - 114514);

    int length;
    char str[5];
    float *array, *res1, *res2;
    if (rank == 0){
        printf("Please input the length of array\n");
        scanf("%d", &length);
        scanf("%s", str);
    }
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(str, 5, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    array = (float *)malloc(length * sizeof(float));
    res1 = (float *)malloc(length * sizeof(float));
    res2 = (float *)malloc(length * sizeof(float));
    memset(res1, 0, length * sizeof(float));
    memset(res2, 0, length * sizeof(float));
    for (int i = 0; i < length; i++){
        array[i] = (float)rand() / RAND_MAX;
    }

    double start_time, end_time;

    // baseline MPI_Allreduce
    start_time = MPI_Wtime();
    if(str[0] == 's')
    	MPI_Allreduce(array, res1, length, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    else
    	MPI_Allreduce(array, res1, length, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    printf("process %d MPI_Allreduce time: %.10lf sec\n", rank, (double)(end_time - start_time));


    for (int i = 0; i < length; i++)
        res2[i] = array[i];
    // ringallreduce
    start_time = MPI_Wtime();
    if(str[0] == 's')
    	RingAllreduce(res2, length, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    else
    	RingAllreduce(res2, length, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    printf("process %d RingAllreduce time: %.10lf sec\n", rank, (double)(end_time - start_time));


    int error_count = 0;
    for (int i = 0; i < length; i++){
        if (fabs(res1[i] - res2[i]) >= 1e-6){
            printf("Error: res_mat1[%d] = %f, res_mat2[%d] = %f\n", i, res1[i], i, res2[i]);
            error_count += 1;
        }
    }
    if (error_count == 0){
        printf("process %d result check all passed!\n", rank);
    }

    free(array);
    free(res1);
    free(res2);
    MPI_Finalize();
    return 0;
}
