#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

/*  This function calculates C = A X B, where
     A is a (m x k) matrix;
     B is a (k x n) matrix;
     C is a (m x n) matrix.
*/

void mat_mat(float *A, float *B, float *C, int m, int k, int n) {
  float sum = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      sum = 0;
      for (int t = 0; t < k; t++) {
        sum += A[i * k + t] * B[t * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

void compare_result(float *mat_1, float *mat_2, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int index = i * n + j;
      if (abs(mat_1[index] - mat_2[index]) > 1e-4) {
        cout << "precision loss is more than 0.0001" << endl;
        return;
      }
    }
  }
  cout << "precision loss is less than 0.0001" << endl;
}

void print_matrix(float *mat, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      cout << mat[i * n + j] << "\t";
    }
    cout << endl;
  }
}

int main(int argc, char **argv) {
  int numprocs, myid;
  int m, k, n;
  float *A, *B, *cpu_C, *mpi_C;
  int srow, srow_last;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  printf("I am proc %d\n", myid);
  // 接收矩阵维数 初始化浮点矩阵
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    m = strtol(argv[1], NULL, 10);
    k = strtol(argv[2], NULL, 10);
    n = strtol(argv[3], NULL, 10);
    cout << "Matrix size : "
         << "m X k : " << m << " X " << k << ", and k X n : " << k << " X " << n
         << endl;

    int A_size = m * k;
    int B_size = k * n;
    int C_size = m * n;
    // 申请矩阵堆存储空间
    A = (float *)malloc(A_size * sizeof(float));
    B = (float *)malloc(B_size * sizeof(float));
    cpu_C = (float *)malloc(C_size * sizeof(float));
    mpi_C = (float *)malloc(C_size * sizeof(float));
    // 初始化浮点矩阵 [0, 100]
    for (int i = 0; i < m * k; i++) {
      A[i] = 0.0 + static_cast<float>(rand()) /
                       (static_cast<float>(RAND_MAX / 100.0));
    }
    for (int i = 0; i < k * n; i++) {
      B[i] = 0.0 + static_cast<float>(rand()) /
                       (static_cast<float>(RAND_MAX / 100.0));
    }

    // CPU计算
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    mat_mat(A, B, cpu_C, m, k, n);
    cpu_end = clock();
    cout << "CPU Elpased Time : "
         << static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0
         << "ms" << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (numprocs == 1) {
    return 0;
  }
  clock_t mpi_start, mpi_end;
  if (myid == 0) {

    mpi_start = clock();
  }
  // 多于一个进程 开始MPI计算
  // 广播维度数据
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("process %d gets k = %d, n=%d\n", myid, k, n);
  if (myid != 0) {
    B = (float *)malloc(k * n * sizeof(float));
  }
  // 广播B矩阵
  MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  srow = m / numprocs;
  // cout << "First srow : " << srow << endl;
  srow_last = m - srow * (numprocs - 1);
  if (myid == numprocs - 1) {
    srow = srow_last;
  }
  if (myid == 0) {
    // cout << 1 << endl;
    int i;
    for (i = 1; i < numprocs - 1; i++) {
      MPI_Send(A + i * srow * k, srow * k, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
    // cout << 2 << endl;
    MPI_Send(A + i * srow * k, srow_last * k, MPI_FLOAT, numprocs - 1, 0,
             MPI_COMM_WORLD);

    // cout << 3 << endl;
    mat_mat(A, B, mpi_C, srow, k, n);
    // print_matrix(cpu_C, m, n);

    // cout << 4 << endl;
    for (i = 1; i < numprocs - 1; i++) {
      MPI_Recv(mpi_C + i * srow * n, srow * n, MPI_FLOAT, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    // cout << 5 << endl;
    MPI_Recv(mpi_C + i * srow * n, srow_last * n, MPI_FLOAT, numprocs - 1, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // print_matrix(A, m, k);
    // print_matrix(B, k, n);
    // print_matrix(cpu_C, m, n);
    // print_matrix(mpi_C, m, n);
    compare_result(cpu_C, mpi_C, m, n);
    // cout << 6 << endl;
    mpi_end = clock();
    cout << "P2P MPI Elpased Time : "
         << static_cast<double>(mpi_end - mpi_start) / CLOCKS_PER_SEC * 1000.0
         << "ms" << endl;
  } else {
    A = (float *)malloc(srow * k * sizeof(float));
    mpi_C = (float *)malloc(srow * n * sizeof(float));
    // print_matrix(B, k, n);
    // cout << "child 1" << endl;
    MPI_Recv(A, srow * k, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // cout << "child 2" << endl;
    mat_mat(A, B, mpi_C, srow, k, n);
    // cout << "child 3" << endl;
    // cout << "srow:" << srow << ", n:" << n << endl;
    // print_matrix(mpi_C, srow, n);
    MPI_Send(mpi_C, srow * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
  free(A);
  free(B);
  if (myid == 0) {
    free(cpu_C);
  }
  free(mpi_C);
  MPI_Finalize();
  return 0;
}