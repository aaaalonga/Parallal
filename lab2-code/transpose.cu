#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <sys/time.h>
#include <stdlib.h>
using namespace std;

const int TILE_WIDTH = 16;

//CPU
void Transpose_CPU(float *h_M, float *h_Mt, int m, int n)
{
    for (size_t j = 0; j < n; j ++)
	{
		for (size_t i = 0; i < m; ++i)
			h_Mt[j * m + i] = h_M[i * n + j];
	}
}

//GPU no shared
__global__ void Transpose_GPU(float *d_M, float *d_M_1, int m, int n)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col < n && row < m)
        d_M_1[col * m + row] = d_M[row * n + col];        
}

//GPU use shared
__global__ void Transpose_GPU_Shared(float *d_M, float *d_M_2, int m, int n)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    ds_M[threadIdx.y][threadIdx.x] = (row < m && col < n) ? d_M[row * n + col] : 0;
    __syncthreads();
    if ((threadIdx.y + blockDim.x * blockIdx.x) < n && (threadIdx.x + blockDim.y * blockIdx.y) < m)
        d_M_2[(threadIdx.y + blockDim.x * blockIdx.x) * m + threadIdx.x + blockDim.y * blockIdx.y] = ds_M[threadIdx.x][threadIdx.y];
}

//Init
void Init(float *h_M, int m, int n)
{
    srand(time(NULL));
    for(int i = 0; i < m * n; ++i)
    {
        h_M[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;		 
    }
}
//Print
void Print(float *h_M, int m, int n)
{
    printf("Matric M:\n");
    for(int i = 0; i < m; i ++)
    {
        for (int j = 0; j < n; j ++)
        {
            printf("%f ",h_M[i * n + j]);
        }
        printf("\n");
    }
}

int main()
{
    int m, n;
    // m = 1000, n = 1000;
    FILE *file1 = fopen("input2.txt","r");
    FILE *file2 = fopen("output2.txt", "w");
    fscanf(file1, "%d,%d,%d", &m, &n);
    float *h_M, *h_Mt, *d_M;
    float *h_M_1, *d_M_1;
    float *h_M_2, *d_M_2;

    size_t sizeM = m * n * sizeof(float);

    h_M = (float *)malloc(sizeM);
    h_Mt = (float *)malloc(sizeM);
    h_M_1 = (float *)malloc(sizeM);
    h_M_2 = (float *)malloc(sizeM);

    cudaMalloc(&d_M, sizeM);
    cudaMalloc(&d_M_1, sizeM);
    cudaMalloc(&d_M_2, sizeM);

    Init(h_M, m, n);
    //Print(h_M, m, n);

    cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);

    dim3 grid((int) ceil(m * 1.0 / TILE_WIDTH), (int) ceil(n * 1.0 / TILE_WIDTH));    
    dim3 block(TILE_WIDTH, TILE_WIDTH); 
    for(int i = 0; i < 3; i ++)
    {
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        if(i == 0)   // CPU
        {
            Transpose_CPU(h_M, h_Mt, m, n);
        } 
        else if (i == 1)   //no shared GPU
        {
            Transpose_GPU<<<grid,block>>>(d_M, d_M_1, m, n);
        } 
        else if (i == 2)    //shared GPU
        {
            Transpose_GPU_Shared<<<grid,block>>>(d_M, d_M_2, m, n);
        }
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float ElapsedTime;
        cudaEventElapsedTime(&ElapsedTime,start,stop);
        
        if(i == 0)   
        {
            printf("Host Elpased Time: %.3f ms\n",ElapsedTime);
            fprintf(file2, "%.3f,", ElapsedTime);
        } 
        else if (i == 1)   
        {
            printf("No Shared Kernel Elpased Time: %.3f ms\n",ElapsedTime);
            fprintf(file2, "%.3f,", ElapsedTime);
            cudaMemcpy(h_M_1, d_M_1, sizeM, cudaMemcpyDeviceToHost);
        }
        else if (i == 2)
        {
            printf("Use Shared Kernel Elpased Time: %.3f ms\n",ElapsedTime);
            fprintf(file2, "%.3f", ElapsedTime);
            cudaMemcpy(h_M_2, d_M_2, sizeM, cudaMemcpyDeviceToHost);
        }
    }

    //Verify
    int flag1 = 0, flag2 = 0;
    for(int i = 0; i < m * n; i ++)
    {
        if(h_Mt[i] != h_M_1[i]) {
            flag1 = 1;
            printf("CPU result and GPU result differ\n");
            printf("%d %d %d\n", i, h_Mt[i], h_M_1[i]);
        }
        if(h_Mt[i] != h_M_2[i]) {
            flag2 = 1;
            printf("CPU result and GPU shared result differ\n");
            printf("%d %d %d\n", i, h_Mt[i], h_M_2[i]);
        }
    }
    if (flag1 == 0)
        printf("CPU result and GPU result same\n");
    if (flag2 == 0)
        printf("CPU result and GPU shared result same\n");

    free(h_M);
    free(h_Mt);
    free(h_M_1);
    free(h_M_2);
    cudaFree(d_M_1);
    cudaFree(d_M_2);
    fclose(file1);
    fclose(file2);

    return 0;
}