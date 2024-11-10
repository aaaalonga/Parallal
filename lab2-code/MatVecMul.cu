#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <sys/time.h>
#include <stdlib.h>
using namespace std;


const int TILE_WIDTH = 16;
// CPU
void MatVecMul_CPU(float *h_M, float *h_N, float *h_C, int m, int n)
{
    for(int i = 0; i < m; i ++)
    {
        float sum = 0.0;
        for (int j = 0; j < n; j ++)
        {
        sum += h_M[i * n + j] * h_N[j];
        }
        h_C[i] = sum;
    }
}

// GPU Global
__global__ void MatVecMul_Global(float *d_M, float *d_N, float *d_P_1, int m, int n)
{
    const size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < m)
	{
		float res = 0; 
		for (size_t j = 0; j < n; ++j)
			res += d_M[i * n + j] * d_N[j];
		d_P_1[i] = res;
	}
}

// GPU Transpose
__global__ void MatVecMul_Transpose(float *d_Mt  , float *d_N, float *d_P_2, int m, int n)
{
    const size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < m)
	{
		float res = 0; 
		for (size_t j = 0; j < n; ++j)
			res += d_Mt[j * m + i] * d_N[j];
		d_P_2[i] = res;
	}
}

// GPU shared
__global__ void MatVecMul_shared(float *d_M, float *d_N, float *d_P_3, int m, int n)
{
    __shared__ float ds_N[TILE_WIDTH];
    
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    float val = 0.0;

    for (size_t t =  0; t < ( n - 1) / TILE_WIDTH + 1; t ++)
    {
        int temp = t * TILE_WIDTH;
        if (temp + threadIdx.y < n)
            ds_N[threadIdx.y] = d_N[temp + threadIdx.y];
        else
            ds_N[threadIdx.y] = 0.0;
        __syncthreads();
        if (row < m)
            for (int i = 0; i < TILE_WIDTH; i ++)
                val += d_M[row * n + i + temp] * ds_N[i];
        __syncthreads();
    }
    if (row < m)
        d_P_3[row] = val;
}

//Init
void Init(float *h_M, float *h_N, int m, int n)
{
    for(int i = 0; i < m * n; ++i)
    {
        h_M[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;		 
    }

    for(int i = 0; i < n; ++i)
    {
        h_N[i] = ((float)rand()/(float)(RAND_MAX)) * 1.0;
    }
}

//Print_Init
void Print(float *h_M, float *h_N, int m, int n)
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
    printf("Vector N:\n");
    for(int i = 0; i < n; ++i)
    {
        printf("%f ", h_N[i]);
    }
    printf("\n");
}

int main()
{
    int m, n;
    FILE *file1 = fopen("input1.txt","r");
    FILE *file2 = fopen("output1.txt", "w");
    fscanf(file1, "%d,%d,%d", &m, &n);

    float *h_M, *h_N, *d_M, *d_N, *h_C;
    float *h_P_1, *d_P_1;
    float *h_P_2, *d_P_2;
    float *h_P_3, *d_P_3;
    float *h_Mt, *d_Mt;

    size_t sizeM = m * n * sizeof(float);
    size_t sizeN = n * sizeof(float);
    size_t sizeP = m * sizeof(float);

    h_M = (float *) malloc(sizeM);
    h_N = (float *) malloc(sizeN);
    h_P_1 = (float *) malloc(sizeP);
    h_P_2 = (float *) malloc(sizeP);
    h_P_3 = (float *) malloc(sizeP);
    h_C = (float *) malloc(sizeP);
    h_Mt = (float *) malloc(sizeM);

    cudaMalloc(&d_M, sizeM);
    cudaMalloc(&d_Mt, sizeM);
    cudaMalloc(&d_N, sizeN);
    cudaMalloc(&d_P_1, sizeP);
    cudaMalloc(&d_P_2, sizeP);
    cudaMalloc(&d_P_3, sizeP);
    
    srand(time(NULL));
    Init(h_M, h_N, m, n);
    //Print(h_M, h_N, m, n);
    for (size_t j = 0; j < n; j ++)
	{
		for (size_t i = 0; i < m; ++i)
			h_Mt[j * m + i] = h_M[i * n + j];
	}
   
    cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mt, h_Mt, sizeM, cudaMemcpyHostToDevice);
          
    dim3 grid(1, (int) ceil(m * 1.0 / TILE_WIDTH));    
    dim3 block(1, TILE_WIDTH);  
    for(int i = 0; i < 4; i ++)
    {
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        if(i == 0)   // CPU
        {
            MatVecMul_CPU(h_M, h_N, h_C, m, n);
        } 
        else if (i == 1)   //global GPU
        {
            MatVecMul_Global<<<grid,block>>>(d_M, d_N, d_P_1, m, n);
        } 
        else if (i == 2)    //transpose GPU
        {
            MatVecMul_Transpose<<<grid,block>>>(d_Mt, d_N, d_P_2, m, n);
        }
        else if (i == 3)     //shared GPU
        {   
            MatVecMul_shared<<<grid,block>>>(d_M, d_N, d_P_3, m, n);
        }  
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float ElapsedTime;
        cudaEventElapsedTime(&ElapsedTime,start,stop);
        
        if(i == 0)   
        {
            printf("Host Elpased Time: %.3f ms\t\tResults: ",ElapsedTime);
            fprintf(file2, "%.3f,", ElapsedTime);
            for(int i = 0; i < (m < 5 ? m : 5); i ++)
            {
                printf("%f ",h_C[i]);
            }
            printf("\n");
        } 
        else if (i == 1)   
        {
            printf("Global Kernel Elpased Time: %.3f ms   \tResults: ",ElapsedTime);
            cudaMemcpy(h_P_1, d_P_1, sizeP, cudaMemcpyDeviceToHost);
            for(int i = 0; i < (m < 5 ? m : 5) ; ++i)
                printf("%f ",h_P_1[i]);
            printf("\n");
        }
        else if (i == 2)
        {
            printf("Transpose Kernel Elpased Time: %.3f ms\tResults: ",ElapsedTime);
            cudaMemcpy(h_P_2, d_P_2, sizeP, cudaMemcpyDeviceToHost);
            for(int i = 0; i < (m < 5 ? m : 5) ; ++i)
                printf("%f ",h_P_2[i]);
            printf("\n");
        }
        else if (i == 3)
        {
            printf("Shared Kernel Elpased Time: %.3f ms\tResults: ",ElapsedTime);
            fprintf(file2, "%.3f", ElapsedTime);
            cudaMemcpy(h_P_3, d_P_3, sizeP, cudaMemcpyDeviceToHost);
            for(int i = 0; i < (m < 5 ? m : 5) ; ++i)
                printf("%f ",h_P_3[i]);
            printf("\n");
        } 
    }
    free(h_P_1);
    free(h_P_2);
    free(h_P_3);
    free(h_M);
    free(h_N);
    free(h_C);
    cudaFree(d_P_1);
    cudaFree(d_P_2);
    cudaFree(d_P_3);
    cudaFree(d_M);
    cudaFree(d_N);
    fclose(file1);
    fclose(file2);

    return 0;
}
