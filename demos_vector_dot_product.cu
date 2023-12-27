#include <iostream>
#define DATATYPE float

void initData(DATATYPE *a, DATATYPE *b, int n)
{
    for(int i=0; i<n; i++)
    {
        a[i]=i/1000.0;
        b[i]=2*i/1000.0;
        // printf("a[i]= %f, b[i]= %f\n", a[i], b[i]);
    }
}

void compare_result(DATATYPE *x, DATATYPE *y, int n)
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
    if(cnt ==0)
        printf("result matched.\n");
    else
        printf("something error! result not match number = %d int total number: %d .\n", cnt, n);

}

void vector_dot_product_serial(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n)
{
    double temp =0.0;
    for(int i=0; i<n; i++)
    {
        temp += a[i]* b[i];
    }
    *c = temp;
}

__global__ void vector_dot_product_gpu_1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n)
{
    const int threadnum=128;
    __shared__ double tmp[threadnum];
    const int tidx = threadIdx.x;
    const int t_n = blockDim.x;
    int tid = tidx;
    double temp =0.0;
    while(tid<n)
    {
        temp += a[tid]* b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();
    
    int i=2, j=1;
    while(i<=threadnum)
    {
        if((tidx%i)==0)
        {
            tmp[tidx] += tmp[tidx+j];
        }
        __syncthreads();
        i *= 2;
        j *= 2;
    }
    if (tidx ==0)
    {
        c[0]=tmp[0];
    }
}

__global__ void vector_dot_product_gpu_2(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n)
{
    const int threadnum=128;
    __shared__ double tmp[threadnum];
    const int tidx =threadIdx.x;
    const int t_n = blockDim.x;
    
    int tid =tidx;
    double temp =0.0;
    while(tid < n)
    {
        temp += a[tid]*b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();

    int i = threadnum/2;
    while(i!=0)
    {
        if(tidx<i)
        {
            tmp[tidx] += tmp[tidx +i];
        }
        __syncthreads();
        i/=2;
    }
    if(tidx ==0)
    {
        c[0]= tmp[0];
    }


}

__global__ void vector_dot_product_gpu_3(DATATYPE *a, DATATYPE *b, DATATYPE *c_temp, int n)
{
    const int threadnum=128;
    __shared__ double tmp[threadnum];
    const int tidx =threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;

    int tid = bidx * blockDim.x + tidx;
    double temp =0.0;
    while(tid <n)
    {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx]= temp;
    __syncthreads();
    
    int i = threadnum /2;
    while(i!=0)
    {
        if(tidx < i)
        {
            tmp[tidx] += tmp[tidx +i];
        }
        __syncthreads();
        i /= 2;
    }
    if(tidx ==0)
    {
        c_temp[bidx]=tmp[0];
    }


}

__global__ void vector_dot_product_gpu_4(DATATYPE *tmp_c, DATATYPE *c)
{
    const int blocknum=32;
    __shared__ double temp[blocknum];

    const int tidx =threadIdx.x;
    temp[tidx]=tmp_c[tidx];
    __syncthreads();
    int i = blockDim.x/2;
    while(i!=0)
    {
        if(tidx <i)
        {
            temp[tidx]+= temp[tidx+i];
        }
        __syncthreads();
        i /=2;
    }

    if(tidx ==0)
    {
        c[i]=temp[0];
    }

}

__global__ void vector_dot_product_gpu_5_0(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n)
{
    if(threadIdx.x ==0 && blockIdx.x ==0)
    {
        c[0]=0.0;
    }
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp =0.0;
    while(tid<n)
    {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    atomicAdd(c, temp);

}

__global__ void vector_dot_product_gpu_5(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n)
{
    if(threadIdx.x ==0 && blockIdx.x ==0)
    {
        c[0]=0.0;
    }
    const int threadnum =128;
    __shared__ double tmp[threadnum];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp =0.0;
    while(tid<n)
    {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = threadnum/2;
    while(i!=0)
    {
        if(tidx <i)
        {
            tmp[tidx] += tmp[tidx +i];
        }
        __syncthreads();
        i/=2;
    }
    if(tidx ==0)
    {
        atomicAdd(c, tmp[0]);
    }

}

int main()
{
    int n=8192;
    DATATYPE a[n], b[n],c[1], h_c1[1],h_c2[1],h_c3[1], h_c5_0[1], h_c5[1];
    initData(a,b,n);

    vector_dot_product_serial(a,b,c,n);

    int threadnum = 128;
    int blocknum = 32;

    DATATYPE *d_a, *d_b, *d_c1, *d_c2,*d_c3, *d_c3_tmp, *d_c5_0, *d_c5;

    cudaMalloc((void**) &d_a, sizeof(DATATYPE)*n);
    cudaMalloc((void**) &d_b, sizeof(DATATYPE)*n);
    cudaMemcpy(d_a, a, sizeof(DATATYPE)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE)*n, cudaMemcpyHostToDevice);


    cudaMalloc((void**) &d_c1, sizeof(DATATYPE)*1);
    // cudaMemcpy(d_c1, c, sizeof(DATATYPE)*1, cudaMemcpyHostToDevice);
    vector_dot_product_gpu_1<<<1, threadnum>>>(d_a, d_b, d_c1, n);
    cudaMemcpy(h_c1, d_c1, sizeof(DATATYPE)*1, cudaMemcpyDeviceToHost);
    cudaFree(d_c1);
    compare_result(c, h_c1, 1);

    cudaMalloc((void**) &d_c2, sizeof(DATATYPE)*1);
    vector_dot_product_gpu_2<<<1, threadnum>>>(d_a, d_b, d_c2, n);
    cudaMemcpy(h_c2, d_c2, sizeof(DATATYPE)*1, cudaMemcpyDeviceToHost);
    cudaFree(d_c2);
    compare_result(c, h_c2, 1);


    cudaMalloc((void**) &d_c3, sizeof(DATATYPE)*1);
    cudaMalloc((void**) &d_c3_tmp, sizeof(DATATYPE)*blocknum);
    vector_dot_product_gpu_3<<<blocknum, threadnum>>>(d_a, d_b, d_c3_tmp, n);
    vector_dot_product_gpu_4<<<1, blocknum>>>(d_c3_tmp, d_c3);
    cudaMemcpy(h_c3, d_c3, sizeof(DATATYPE)*1, cudaMemcpyDeviceToHost);
    cudaFree(d_c3);
    compare_result(c, h_c3, 1);

    cudaMalloc((void**) &d_c5_0, sizeof(DATATYPE)*1);
    vector_dot_product_gpu_5_0<<<blocknum, threadnum>>>(d_a, d_b, d_c5_0, n);
    cudaMemcpy(h_c5_0, d_c5_0, sizeof(DATATYPE)*1, cudaMemcpyDeviceToHost);
    cudaFree(d_c5_0);
    compare_result(c, h_c5_0, 1);

    cudaMalloc((void**) &d_c5, sizeof(DATATYPE)*1);
    vector_dot_product_gpu_5<<<blocknum, threadnum>>>(d_a, d_b, d_c5, n);
    cudaMemcpy(h_c5, d_c5, sizeof(DATATYPE)*1, cudaMemcpyDeviceToHost);
    cudaFree(d_c5);
    compare_result(c, h_c5, 1);


    cudaFree(d_a);
    cudaFree(d_b);
    

}
