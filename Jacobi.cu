#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void fill_random(float *A, int N, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //A[j + N * i] = i + j;
    float *row = (float *)((char *) A + i * pitch);
    row[j] = i + j;
}

int main(int argc, char **argv) {
    int N = 10;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    std::clog <<  "Current matrix size: " << N << std::endl;

    cudaSetDevice(0);
    float *A = new float[N * N];
    float *f = new float[N];
    float *x = new float[N];
    for (int i = 0; i < N; i++) {
        f[i] = i;
        x[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 0;
        }
    }
    float *ADev, *fDev, *xDev;
    size_t pitchA;
    cudaMallocPitch((void **) &ADev, &pitchA, N * sizeof(float), N);
    cudaMalloc((void **) &fDev, N * sizeof(float));
    cudaMalloc((void **) &xDev, N * sizeof(float));
    cudaMemcpy2D(ADev, pitchA, A, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy(fDev, f, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xDev, x, N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    fill_random<<<numBlocks, threadsPerBlock>>>(ADev, N, pitchA);
    
    //cudaMemcpy(A, ADev, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy2D(A, N * sizeof(float), ADev, pitchA, N * sizeof(float), N, cudaMemcpyDeviceToHost);
    cudaMemcpy(f, fDev, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x, xDev, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + i] += A[i * N + j];
        }
    }
    cudaFree(ADev);
    cudaFree(fDev);
    cudaFree(xDev);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4.0f ", A[i * N + j]);
        }
        printf("| %4.0f\n", f[i]);
    }
    delete A;
    delete f;
    delete x;
    return 0;
}