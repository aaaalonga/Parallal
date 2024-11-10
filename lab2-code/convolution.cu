 #include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#define TILE_WIDTH 16
using namespace std;
float cpu_time;

// init M
void init_M(int *h_M, int m, int n)
{
    for(int i = 0; i < m * n; i++)
        h_M[i] = rand() % 255;
}
// init K
void init_K(int *Conv, int k)
{
    for(int i = 0; i < k * k; i++)
        Conv[i] = rand() % 5;
}

void Conv_CPU(int *h_M, int *Conv, int *h_M_1, int m, int n, int k)
{
    int p = m - k + 1;
    int q = n - k + 1;
    for(int i = 0; i < p; i ++)
        for(int j = 0; j < q; j ++)
        {
            int sum = 0;
            for(int a = 0;  a < k; a ++)
                for(int b = 0; b < k; b ++)
                    sum += h_M[(i + a) * n + b + j] * Conv[a * k + b];
            h_M_1[i * q + j] = sum;
        }
    
}
__global__ void ConvSharedKernel(int *d_M, int *d_Conv, int *d_M_2, int m, int n, int k)
{
    __shared__ int ds_Conv[TILE_WIDTH][TILE_WIDTH];
    int p = m - k + 1;
    int q = n - k + 1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    ds_Conv[ty][tx] = (tx < k && ty < k) ? d_Conv[ty * k + tx] : 0;
    __syncthreads();
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < p && col < q)
    {
        int sum = 0;
        for (int i = 0; i < k; i ++)
        {
            for (int j = 0; j < k; j ++)
            {
                sum += ds_Conv[i][j] * d_M[(row + i) * n + col + j];
            }
        }
        d_M_2[row * q + col] = sum;
    }
}

int main()
{
    int k, m, n;
    FILE *file1 = fopen("input3.txt","r");
    FILE *file2 = fopen("output3.txt", "w");
    fscanf(file1, "%d,%d,%d", &m, &n, &k);
    int p = m - k + 1;
    int q = n - k + 1;

    int *h_M, *h_M_1, *h_M_2;
    int *Conv, *d_Conv;
    int *d_M, *d_M_2;

    size_t sizeM = m * n * sizeof(int);
    size_t sizeK = k * k * sizeof(int);
    size_t sizeP = p * q * sizeof(int);

    h_M = (int *) malloc(sizeM);
    Conv = (int *) malloc(sizeK);
    h_M_1 = (int *) malloc(sizeP);
    h_M_2 = (int *) malloc(sizeP);

    cudaMalloc(&d_M, sizeM);
    cudaMalloc(&d_M_2, sizeP);
    cudaMalloc(&d_Conv, sizeK);

    init_M(h_M, m, n);
    init_K(Conv, k);

    // for (int i = 0; i < m; i ++)
    // {
    //     for (int j = 0; j < n; j ++)
    //     {
    //         printf("%d ", h_M[i * n + j]);
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < k; i ++)
    // {
    //     for (int j = 0; j < k; j ++)
    //     {
    //         printf("%d ", Conv[i * k + j]);
    //     }
    //     printf("\n");
    // }

    cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Conv, Conv, sizeK, cudaMemcpyHostToDevice);
    
    dim3 grid((int) ceil(p * 1.0 / TILE_WIDTH), (int) ceil(q * 1.0 / TILE_WIDTH));
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    for(int i = 0; i < 2; i ++)
    {
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        if(i == 0)   // CPU
        {
            Conv_CPU(h_M, Conv, h_M_1, m, n, k);
        } 
        else if (i == 1)   //GPU
        {
            ConvSharedKernel<<<grid,block>>>(d_M, d_Conv, d_M_2, m, n, k);
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
            printf("Kernel Elpased Time: %.3f ms\n",ElapsedTime);
            fprintf(file2, "%.3f", ElapsedTime);
            cudaMemcpy(h_M_2, d_M_2, sizeP, cudaMemcpyDeviceToHost);
        }
    }
    
    int flag = 0;
    for(int i = 0; i < p * q; i ++)
        if(h_M_1[i] != h_M_2[i]) {
            flag = 1;
            printf("CPU result and GPU shared result differ\n");
            printf("%d %d %d\n", i, h_M_1[i], h_M_2[i]);
        }
    if (flag == 0)
        printf("CPU result and GPU shared result same\n");
    // for (int i = 0; i < p; i ++)
    // {
    //     for (int j = 0; j < q; j ++)
    //     {
    //         printf("%d ", h_M_1[i * q + j]);
    //     }
    //     printf("\n");
    // }
    free(Conv);
    free(h_M);
    free(h_M_1);
    free(h_M_2);
    cudaFree(d_M);
    cudaFree(d_M_2);
    cudaFree(d_Conv);
    fclose(file1);
    fclose(file2);
    return 0;

}

