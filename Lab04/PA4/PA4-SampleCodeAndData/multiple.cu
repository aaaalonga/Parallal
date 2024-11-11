#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>



void SequentialCalculation(const int &n,
                           const int &m,
                           const std::vector<std::vector<int>> &A,
                           const std::vector<std::vector<int>> &B,
                           std::vector<std::vector<int>> *C) {

  std::vector<std::vector<int>> B_power, next_B_power;
  std::vector<std::vector<int>> D;
  (*C) = A;
  B_power = B;
  int tmp;
  for (int t = 1; t<=m; t++) {
    D = std::vector<std::vector<int>>(n, std::vector<int>(n,0));
    for (int i = 0; i<n; i++) {
      for (int j = 0; j<n; j++) {
        for (int k = 0; k<n; k++) {
          D[i][j] = (D[i][j] + A[i][k] * B_power[k][j])%2;
        }
      } 
    }
    for (int i = 0; i<n; i++) {
      for (int j = 0; j<n; j++) {
        (*C)[i][j] = ((*C)[i][j] + D[i][j]) %2; 
      }
    } 
    if (t==m)
      break;
    next_B_power = std::vector<std::vector<int>>(n, std::vector<int>(n,0));
    for (int i = 0; i<n; i++) {
      for (int j = 0; j<n; j++) {
        for (int k = 0; k<n; k++)
          next_B_power[i][j] = (next_B_power[i][j]+ B_power[i][k]*B[k][j])%2;
      } 
    }
    B_power = next_B_power;
  }
}

bool LoadFile(const std::string &input_file_path, int *n, int *m, std::vector<std::vector<int>> *A,
              std::vector<std::vector<int>> *B) {
  std::ifstream fin(input_file_path.c_str());
  if (!fin.is_open()) {
    return false;
  }
  fin >> (*n) >> (*m);
  *A = std::vector<std::vector<int>>(*n,std::vector<int>(*n,0));
  *B = std::vector<std::vector<int>>(*n,std::vector<int>(*n,0));
  for (int i = 0;i < (*n); i++)
    for (int j = 0;j < (*n); j++)
      fin >> (*A)[i][j];
  for (int i = 0;i < (*n); i++)
    for (int j = 0;j < (*n); j++)
      fin >> (*B)[i][j];
  fin.close();
  return true;
}

void TestAnswerCorrectness(const std::vector<std::vector<int>> &sequential_answer,
                           const std::vector<std::vector<int>> &parallel_answer) {
  if (sequential_answer.size() != parallel_answer.size()) {
    std::cout << "Error! The number of sequential_answer and parallel_answer "
                 "is not the same"
              << std::endl;
    return ;
  }
  long long sum_sequential_answer = 0;
  long long sum_parallel_answer = 0;
  int sum_error = 0;
  for (uint i = 0; i < sequential_answer.size(); i++) {
    if (sequential_answer[i].size() != parallel_answer[i].size())
    {
      std::cout << "Error! The number of sequential_answer and parallel_answer "
                 "is not the same"
              << std::endl;
      return ;
    }
    for (uint j = 0; j < sequential_answer[i].size(); j++) {
      sum_error +=  abs(sequential_answer[i][j] - parallel_answer[i][j]);
      sum_sequential_answer += sequential_answer[i][j];
      sum_parallel_answer += parallel_answer[i][j];  
    }
  }
  std::cout << "sum_sequential_answer = " << sum_sequential_answer << std::endl;
  std::cout << "sum_parallel_answer = " << sum_parallel_answer << std::endl;

  if (sum_error > 0) {
    std::cout << "Wrong Answer" << std::endl;
  } else {
    std::cout << "Correct!!!" << std::endl;
  }
}

// ==============================================================
// ====    Write your functions below this line    ====
// ==============================================================
// ==============================================================




// ==============================================================
// ====    Write your functions above this line    ====
// ==============================================================
// ==============================================================


int main(int argc, char **argv) {
  int number_of_processes, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double parallel_start_time;

  int number_of_block_in_a_grid;
  int number_of_thread_in_a_block;
  int n,m;
  std::vector<std::vector<int>> A;
  std::vector<std::vector<int>> B;
  if (rank == 0) {
    if (argc < 4) {
      std::cout << "Error! Please use \"mpiexec -n [process number] "
                   "[--hostfile hostfile] multiple [number_of_block_in_a_grid] [number_of_thread_in_a_block] [data_file_name]\"\n";
      return 1;
    } else {
      number_of_block_in_a_grid = std::atoi(argv[1]);
      number_of_thread_in_a_block = std::atoi(argv[2]);
      std::string input_file_path = std::string(argv[3]);
      std::cout << "number_of_block_in_a_grid:" << number_of_block_in_a_grid<< std::endl;
      std::cout << "number_of_thread_in_a_block:" << number_of_thread_in_a_block<< std::endl;
      if (!LoadFile(input_file_path, &n, &m, &A, &B)) {
        std::cout << "Error! Please check the format of input file\n";
        return 1;
      }
    }
  }
  std::vector<std::vector<int>> parallel_answer;

  if (rank == 0) {
    parallel_start_time = MPI_Wtime();
  }
  
  // ==============================================================
  // ====    Write your implementation below this line    ====
  // ==============================================================
  // ==============================================================




  // ==============================================================
  // ====    Write your implementation above this line    ====
  // ==============================================================
  // ==============================================================
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    double parallel_end_time = MPI_Wtime();
    double parallel_running_time = parallel_end_time - parallel_start_time;
    std::cout << "parallel running time:" << parallel_running_time << std::endl;
    std::vector<std::vector<int>> sequential_answer;
    double sequential_start_time = MPI_Wtime();

    SequentialCalculation(n, m, A, B, &sequential_answer);
    double sequential_end_time = MPI_Wtime();
    double sequential_running_time =
        sequential_end_time - sequential_start_time;
    std::cout << "sequential running time:" << sequential_running_time
              << std::endl;
    std::cout << "speed up:" <<  sequential_running_time/parallel_running_time
              << std::endl;
    TestAnswerCorrectness(sequential_answer, parallel_answer);
  }
  MPI_Finalize();
  return 0;
}