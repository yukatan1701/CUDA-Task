#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define BLOCK_SIZE 16

__global__ void fill_random(float *A, int N, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //A[j + N * i] = i + j;
    float *row = (float *)((char *) A + i * pitch);
    row[j] = i + j;
}

void mult(float *A, float *B, float *C, size_t N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

__global__ void matMult(float *A, size_t pitchA, float *B, size_t pitchB,
                        float *C, size_t pitchC, size_t N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = N * BLOCK_SIZE * by;
    int bBegin = BLOCK_SIZE * bx;
    int aEnd = aBegin + N - 1;

    int aStep = BLOCK_SIZE;
    int bStep = BLOCK_SIZE * N;

    float sum = 0.0f;

    float *rowA = (float *)((char *) A + pitchA * BLOCK_SIZE * by + pitchA * ty);
    float *rowB = (float *)((char *) B + sizeof(float) * BLOCK_SIZE * bx + pitchB * ty);
    
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
    
        as[ty][tx] = rowA[tx];
        // A[ia + N * ty + tx] = [глобальный индекс начала + индекс строки + столбец]
        bs[ty][tx] = rowB[tx];
        rowA = (float *)((char *) rowA + BLOCK_SIZE * sizeof(float));
        rowB = (float *)((char *) rowB + BLOCK_SIZE * pitchB);

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += as[ty][k] * bs[k][tx];
        }
        __syncthreads();
    }

    int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    float *rowC = (float *)((char *) C + pitchC * BLOCK_SIZE * by + BLOCK_SIZE * bx * sizeof(float) + pitchC * ty);
    if (abs(sum - 3160) < 0.001f) {
        printf("HERE\n");
    }
    rowC[tx] = sum;
}

int main(int argc, char **argv) {
    int N = BLOCK_SIZE;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    std::clog << "Given matrix size: " << N << std::endl;

    /// Round up to the next power of 2 (min is 16).
    N = N < BLOCK_SIZE ? BLOCK_SIZE : pow(2, ceil(log2(N)));
    
    std::clog <<  "Current matrix size: " << N << std::endl;

    cudaSetDevice(0);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *A = new float[N * N];
    float *C = new float[N * N];
    float *C2 = new float[N * N];
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
    float *ADev,  *CDev, *fDev, *xDev;
    size_t pitchA, pitchC;
    cudaMallocPitch((void **) &ADev, &pitchA, N * sizeof(float), N);
    cudaMallocPitch((void **) &CDev, &pitchC, N * sizeof(float), N);
    cudaMalloc((void **) &fDev, N * sizeof(float));
    cudaMalloc((void **) &xDev, N * sizeof(float));
    cudaMemcpy2D(ADev, pitchA, A, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy(fDev, f, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xDev, x, N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    printf("Threads per block: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Num blocks: (%d, %d)\n", numBlocks.x, numBlocks.y);

    fill_random<<<numBlocks, threadsPerBlock>>>(ADev, N, pitchA);
    matMult<<<numBlocks, threadsPerBlock>>>(ADev, pitchA, ADev, pitchA, CDev, pitchC, N);

    cudaMemcpy2D(A, N * sizeof(float), ADev, pitchA, N * sizeof(float), N, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(C, N * sizeof(float), CDev, pitchC, N * sizeof(float), N, cudaMemcpyDeviceToHost);
    cudaMemcpy(f, fDev, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x, xDev, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
/*
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + i] += A[i * N + j];
        }
    }*/
    cudaFree(ADev);
    cudaFree(CDev);
    cudaFree(fDev);
    cudaFree(xDev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    mult(A, A, C2, N);

    printf("Elapsed time: %.2f ms\n", gpuTime);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%3.0f ", A[i * N + j]);
        }
        printf("| %3.0f\n", f[i]);
    }
    printf("multiply:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%3.0f ", C[i * N + j]);
        }
        printf("\n");
    }
    printf("right:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(C[i * N + j] == C2[i * N + j]);
            printf("%3.0f ", C2[i * N + j]);
        }
        printf("\n");
    }
    delete A;
    delete f;
    delete x;
    return 0;
}