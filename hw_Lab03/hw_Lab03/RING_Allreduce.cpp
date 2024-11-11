#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

using namespace std;

inline void mpi_max(float *a, float *b) { *a = max(*a, *b); }
inline void mpi_sum(float *a, float *b) { *a += *b; }
typedef void (*fun_ptr)(float *, float *);

void print_arr(float *arr, int n) {
  for (int i = 0; i < n; i++) {
    cout << arr[i] << "\t";
  }
  cout << endl;
}

void reduce(float *dst, float *src, int n, MPI_Op op) {
  if (op == MPI_SUM) {
    for (int i = 0; i < n; i++) {
      // mpi_sum(dst + i, src + i);
      dst[i] += src[i];
    }
  } else if (op == MPI_MAX) {
    for (int i = 0; i < n; i++) {
      // mpi_max(dst + i, src + i);
      dst[i] = dst[i] > src[i] ? dst[i] : src[i];
    }
  }
}

void compare_result(float *res1, float *res2, int n) {
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (abs(res1[i] - res2[i]) > 1e-4) {
      cout << "precision loss is more than 0.0001" << endl;
      return;
      // count++;
    }
  }
  // cout << "count : " << count << endl;
  cout << "precision loss is less than 0.0001" << endl;
}

void ring_based_allreduce(float *arr, int n, int myid, int numprocs,
                          fun_ptr fun) {
  MPI_Request req[2];
  int data_size = n / numprocs;
  int offset = n - myid * data_size;
  float *arr_end = arr + n;

  int pre_pid = (myid == 0) ? numprocs - 1 : myid - 1;
  int next_pid = (myid + 1) % numprocs;
  float *recv = (float *)malloc(data_size * sizeof(float));

  for (int i = 0; i < 2 * (numprocs - 1); i++) {
    float *send = arr_end - offset;
    offset = (offset == n) ? data_size : offset + data_size;
    float *reduce_local = arr_end - offset;

    MPI_Isend(send, data_size, MPI_FLOAT, next_pid, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Recv(recv, data_size, MPI_FLOAT, pre_pid, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    if (i < numprocs - 1) {
      for (int k = 0; k < data_size; k++) {
        fun(reduce_local + k, recv + k);
      }
    } else {
      for (int k = 0; k < data_size; k++) {
        reduce_local[k] = recv[k];
      }
    }
  }
  free(recv);
}

void RING_Allreduce(const void *buf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int numprocs, myid;
  MPI_Comm_size(comm, &numprocs);
  MPI_Comm_rank(comm, &myid);
  // fun_ptr fun;
  // if (op == MPI_SUM) {
  //   fun = mpi_sum;
  // } else if (op == MPI_MAX) {
  //   fun = mpi_max;
  // }
  MPI_Request req[2];
  int data_size = count / numprocs;
  int offset = count - myid * data_size;
  float *buf_end = (float *)buf + count;
  float *recv_end = (float *)recvbuf + count;

  // memcpy(recv_end - offset, buf_end - offset, data_size * sizeof(float));

  int pre_pid = (myid == 0) ? numprocs - 1 : myid - 1;
  int next_pid = (myid + 1) % numprocs;

  for (int i = 0; i < 2 * (numprocs - 1); i++) {
    float *send = (i == 0) ? buf_end - offset : recv_end - offset;
    offset = (offset >= count) ? data_size : offset + data_size;
    float *reduce_send = buf_end - offset;
    float *recv = recv_end - offset;

    MPI_Isend(send, data_size, datatype, next_pid, 0, comm, &req[0]);
    MPI_Irecv(recv, data_size, datatype, pre_pid, 0, comm, &req[1]);
    MPI_Wait(&req[1], MPI_STATUS_IGNORE);

    if (i < numprocs - 1) {
      // for (int k = 0; k < data_size; k++) {
      // fun(recv + k, reduce_send + k);
      // }
      reduce(recv, reduce_send, data_size, op);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char **argv) {
  int numprocs, myid;
  int n;
  float *arr, *mpi_res, *ring_res, *st_ring_res;
  double mpi_start, mpi_end;
  double my_mpi_start, my_mpi_end;
  char *op;
  string op_s;
  MPI_Op ring_op;
  fun_ptr fun;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  op = (char *)malloc(32 * sizeof(char));
  if (myid == 0) {
    strcpy(op, argv[1]);
    n = strtol(argv[2], NULL, 10);
    // cout << "op : " << op << " , n : " << n << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(op, 32, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // cout << "op : " << op << " , n : " << n << endl;
  arr = (float *)malloc(n * sizeof(float));
  mpi_res = (float *)malloc(n * sizeof(float));
  ring_res = (float *)malloc(n * sizeof(float));
  st_ring_res = (float *)malloc(n * sizeof(float));

  srand(static_cast<unsigned int>(myid * 10));
  for (int i = 0; i < n; i++) {
    ring_res[i] = arr[i] = 0.0 + static_cast<float>(rand()) /
                                     (static_cast<float>(RAND_MAX / 100.0));
    // ring_res[i] = arr[i] = 1;
  }
  // cout << "myid : " << myid << ", array : " << endl;
  // print_arr(arr, n);
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    mpi_start = MPI_Wtime();
  op_s = op;
  // cout << op_s << endl;
  if (op_s == "sum") {
    fun = mpi_sum;
    ring_op = MPI_SUM;
    MPI_Allreduce(arr, mpi_res, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  } else if (op_s == "max") {
    fun = mpi_max;
    ring_op = MPI_MAX;
    MPI_Allreduce(arr, mpi_res, n, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  }
  // cout << "myid : " << myid << ", reduction result : " << endl;
  // print_arr(mpi_res, n);
  if (myid == 0) {
    mpi_end = MPI_Wtime();
    // cout << "origin arr : " << endl;
    // print_arr(arr, n);

    // cout << "MPI_Allreduce result : " << endl;
    // print_arr(mpi_res, n);
    cout << "MPI_Allreduce Elpased Time : " << (mpi_end - mpi_start) * 1000
         << " ms" << endl;
  }

  // ring_based_allreduce
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    my_mpi_start = MPI_Wtime();
  ring_based_allreduce(ring_res, n, myid, numprocs, fun);
  if (myid == 0) {
    my_mpi_end = MPI_Wtime();
    // cout << "RING_Allreduce result : " << endl;
    // print_arr(ring_res, n);
    cout << "RING_Allreduce Elpased Time : "
         << (my_mpi_end - my_mpi_start) * 1000 << " ms" << endl;
    compare_result(mpi_res, ring_res, n);
  }

  // standard ring_based_allreduce
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    my_mpi_start = MPI_Wtime();
  RING_Allreduce(arr, st_ring_res, n, MPI_FLOAT, ring_op, MPI_COMM_WORLD);
  if (myid == 0) {
    my_mpi_end = MPI_Wtime();
    // cout << "Standard interface RING_Allreduce result : " << endl;
    // print_arr(st_ring_res, n);
    cout << "Standard interface RING_Allreduce Elpased Time : "
         << (my_mpi_end - my_mpi_start) * 1000 << " ms" << endl;
    compare_result(mpi_res, st_ring_res, n);
  }

  free(arr);
  free(mpi_res);
  free(ring_res);
  free(st_ring_res);
  free(op);
  MPI_Finalize();
  return 0;
}