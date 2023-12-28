#include <iostream>
#include <chrono>

using namespace std;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void initDate(float *arr,int Len, bool randFlag=true)
{
    if (randFlag)
    {
        for (int i = 0; i < Len; i++) {
            arr[i] = rand()/1000000;
        }
    }
    else
    {
        float value =0.0;
        for (int i = 0; i < Len; i++) {
            arr[i] = value;
        }
    }  
}

void compare_result(float *x, float *y, int n, char *name)
{
    int cnt=0;
    for (int i=0; i<n; i++)
    {
        if (x[i]!=y[i])
        {
            cnt++;
            printf("x= %f, y= %f\n", x[i],y[i]);
        }
            
    }
    printf("%s, ", name);
    if(cnt ==0)
        printf("result matched.\n");
    else
        printf("something error! result not match number = %d int total number: %d .\n", cnt, n);

}


void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) 
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

void cpuSgemm_1(float *a, float *b, float *c, const int M, const int N, const int K) 
{
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++)
            {
                c[OFFSET(m, n, N)] += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }           
        }
    }
}

void cpuSgemm_2(float *a, float *b, float *c, const int M, const int N, const int K) 
{
    float* b1=(float*) malloc(sizeof(float)*K*N);
    for(int i=0; i<K; i++)
    {
        for (int j=0; j<N; j++)
        {
            b1[OFFSET(j,i,K)]= b[OFFSET(i,j,N)];
        }
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b1[OFFSET(n, k, K)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}




void operation(void (*func)(float*,float*, float*, int, int, int), float *a, float *b, float *c, const int M, const int N, const int K, int repeat, char* name)
{
    auto begin0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<repeat; i++)
    {
        (*func)(a,b,c, M, N, K);
    }
    auto end0 = std::chrono::high_resolution_clock::now();
    auto elapsed0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0);
    printf("%s, Time measured: %d microseconds.\n", name, int(elapsed0.count()/repeat));
}

__global__ void matrix_multiply_gpu_0(float*a, float*b, float*c, int M, int N, int K)
{
    int tidx =threadIdx.x;
    int bidx = blockIdx.x;
    int idx = bidx * blockDim.x +tidx;
    int row = idx/N;
    int col = idx%N;
    if(row<M && col < N)
    {
        float tmp =0.0;
        for(int k=0; k<K; k++)
        {
            tmp+=a[row*K+k] * b[k*N+col];
        }
        c[row*N+col] = tmp;
    }
}

__global__ void matrix_multiply_gpu_1(float*a, float*b, float*c, int M, int N, int K)
{
    int tidx =threadIdx.x;
    int bidx = blockIdx.x;

    float tmp;
    for(;bidx<M; bidx += gridDim.x)
    {
        for(;tidx<N; tidx+=blockDim.x )
        {
            tmp=0.0;
            for(int k=0; k<K; k++)
            {
                tmp+=a[bidx*K +k] * b[k*N+tidx];
            }
            c[bidx*N+tidx] = tmp;
        }              
    }
}

#define TILE_WIDTH 256
__global__ void matrix_multiply_gpu_2(float*a, float*b, float*c, int M, int N, const int K)
{
    __shared__ float data[TILE_WIDTH];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int i,j;
    for(i=tid; i<K; i+=blockDim.x)
    {
        data[i]=a[row*K +i];
    }
    __syncthreads();
    float tmp;
    for(j=tid; j<N; j+=blockDim.x)
    {
        tmp=0.0;
        for(int k=0; k<K; k++)
        {
            tmp += data[k]*b[k*N+j];
        }
        c[row*N+j] = tmp;
    }
}

#define TILE_SIZE 32
__global__ void matrix_multiply_gpu_3(float*a, float*b, float*c, int M, int N, const int K)
{
    __shared__ float matA[TILE_SIZE][TILE_SIZE];
	__shared__ float matB[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
    int Col = bx * TILE_SIZE + tx;
	int Row = by * TILE_SIZE + ty;
	
	float Pervalue = 0.0;
	for(int i = 0;i < K / TILE_SIZE;i++)  
	{
		matA[ty][tx] = a[Row * K + (i * TILE_SIZE + tx)];
		matB[ty][tx] = b[Col + (i * TILE_SIZE + ty) * N];
		__syncthreads();
	
		for(int k = 0;k < TILE_SIZE;k++) 
			Pervalue += matA[ty][k] * matB[k][tx];
		__syncthreads();
	}
	
	c[Row * N + Col] = Pervalue;
    
}
 

int main()
{
    int M=512;
    int N=512;
    int K=256;

    float *a = (float*) malloc(M*K * sizeof(float));
    float *b = (float*) malloc(N*K * sizeof(float));
    float *c = (float*) malloc(M*N * sizeof(float));
    float *c1 = (float*) malloc(M*N * sizeof(float));
    float *c2 = (float*) malloc(M*N * sizeof(float));
    float *c_gpu_0 = (float*) malloc(M*N * sizeof(float));
    float *c_gpu_1 = (float*) malloc(M*N * sizeof(float));
    float *c_gpu_2 = (float*) malloc(M*N * sizeof(float));
    float *c_gpu_3 = (float*) malloc(M*N * sizeof(float));

    initDate(a,M*K);
    initDate(b,N*K);
    initDate(c, M*N, false);
    initDate(c1, M*N, false);
    initDate(c2, M*N, false);
    initDate(c_gpu_0, M*N, false);
    initDate(c_gpu_1, M*N, false);
    initDate(c_gpu_2, M*N, false);
    initDate(c_gpu_3, M*N, false);

    //ensure result is right.
    cpuSgemm(a,b,c,M,N,K);
    cpuSgemm_1(a,b,c1,M,N,K);
    cpuSgemm_2(a,b,c2,M,N,K); 
    compare_result(c, c1, M*N,"sgemm1");
    compare_result(c, c2,  M*N,"sgemm2");



    //test the prerformance.
    int repeat =10;
    operation(cpuSgemm,a,b,c,M,N,K,repeat,"cpuSgemm");
    operation(cpuSgemm_1,a,b,c1,M,N,K,repeat,"cpuSgemm_1");
    operation(cpuSgemm_2,a,b,c2,M,N,K,repeat,"cpuSgemm_2");
    
    float* d_a, *d_b, *d_c0, *d_c1, *d_c2, *d_c3;
    cudaMalloc((void**) &d_a, sizeof(float)*(M*K));
    cudaMalloc((void**) &d_b, sizeof(float)*(N*K));
    cudaMalloc((void**) &d_c0, sizeof(float)*(M*N));
    cudaMalloc((void**) &d_c1, sizeof(float)*(M*N));
    cudaMalloc((void**) &d_c2, sizeof(float)*(M*N));
    cudaMalloc((void**) &d_c3, sizeof(float)*(M*N));

    cudaMemcpy(d_a, a, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N*K, cudaMemcpyHostToDevice);
    

    int threadnum=64;
    int blocks =(M*N+threadnum-1)/threadnum;
    cudaMemcpy(d_c0, c_gpu_0, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    matrix_multiply_gpu_0<<<blocks, threadnum>>>(d_a, d_b, d_c0, M, N, K);
    cudaMemcpy(c_gpu_0, d_c0, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    compare_result(c, c_gpu_0,  M*N,"gpu_0");
    cudaFree(d_c0);

    cudaMemcpy(d_c1, c_gpu_1, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    matrix_multiply_gpu_1<<<M, threadnum>>>(d_a, d_b, d_c1, M, N, K);
    cudaMemcpy(c_gpu_1, d_c1, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    compare_result(c, c_gpu_1,  M*N,"gpu_1");
    cudaFree(d_c1);

    cudaMemcpy(d_c2, c_gpu_2, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    matrix_multiply_gpu_2<<<M, threadnum>>>(d_a, d_b, d_c2, M, N, K);
    cudaMemcpy(c_gpu_2, d_c2, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    compare_result(c, c_gpu_2,  M*N,"gpu_2");
    cudaFree(d_c2);

    threadnum=32;
    dim3 gridSize(M / threadnum,N / threadnum);
	dim3 blockSize(threadnum,threadnum);
    cudaMemcpy(d_c3, c_gpu_3, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    matrix_multiply_gpu_3<<<gridSize, blockSize>>>(d_a, d_b, d_c3, M, N, K);
    cudaMemcpy(c_gpu_3, d_c3, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    compare_result(c, c_gpu_3,  M*N,"gpu_3");
    cudaFree(d_c3);


    free(a);
    free(b);
    free(c);
    free(c1);
    free(c2);
    free(c_gpu_0);
    free(c_gpu_1);
    free(c_gpu_2);
    free(c_gpu_3);
    cudaFree(d_a);
    cudaFree(d_b);

}