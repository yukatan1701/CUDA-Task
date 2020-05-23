#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <limits>
#define EPS 0.00001f

#define BLOCK_SIZE 4

void printMatrix(float *A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }
}

void printMatrixDev(float *ADev, size_t pitch, int N) {
    float *A = new float[N * N];
    cudaMemcpy2D(A, N * sizeof(float), ADev, pitch, N * sizeof(float), N, cudaMemcpyDeviceToHost);
    printMatrix(A, N);
    delete A;
}

void printVector(float *v, int N) {
    printf("[");
    for (int i = 0; i < N; i++) {
        printf("%.7f ", v[i]);
    }
    printf("]\n");
}

void printVectorDev(float *vDev, int N) {
    float *v = new float[N];
    cudaMemcpy(v, vDev, N * sizeof(float), cudaMemcpyDeviceToHost);
    printVector(v, N);
    delete v;
}

__global__ void fillRandom(float *A, int N, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //A[j + N * i] = i + j;
    float *row = (float *)((char *) A + i * pitch);
    row[j] = i + j + 1;
    atomicAdd(&row[i], row[j]);
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
        bs[ty][tx] = rowB[tx];

        rowA = (float *)((char *) rowA + BLOCK_SIZE * sizeof(float));
        rowB = (float *)((char *) rowB + BLOCK_SIZE * pitchB);

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += as[ty][k] * bs[k][tx];
        }
        __syncthreads();
    }

    float *rowC = (float *)((char *) C + pitchC * BLOCK_SIZE * by +
        BLOCK_SIZE * bx * sizeof(float) + pitchC * ty);
    rowC[tx] = sum;
}

void multMatVecClassic(float *A, float *b, float *c, size_t N) {
    for (int i = 0; i < N; i++) {
        c[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            c[i] += A[i * N + j] * b[j];
        }
    }
}

// C = A - B
__global__ void matDiff(float *A, size_t pitchA, float *B, size_t pitchB,
                        float *C, size_t pitchC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float *rowA = (float *)((char *) A + i * pitchA);
    float *rowB = (float *)((char *) B + i * pitchB);
    float *rowC = (float *)((char *) C + i * pitchC);
    rowC[j] = rowA[j] - rowB[j];
}

// C = D * A, where D is diagonal matrix
__global__ void multDiagMatAndMat(float *D, size_t pitchD, float *A, size_t pitchA,
                                  float *C, size_t pitchC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float *rowA = (float *)((char *) A + i * pitchA);
    float *rowD = (float *)((char *) D + i * pitchD);
    float *rowC = (float *)((char *) C + i * pitchC);
    rowC[j] = rowA[j] * rowD[i];
}

// A * b + g = c
__global__ void multMatVecPlusVec(float *A, size_t pitch, float *b, float *g, float *c, size_t N) {
    //int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = N * BLOCK_SIZE * by;
    int aEnd = aBegin + N - 1;

    int aStep = BLOCK_SIZE;
    
    float *rowA = (float *)((char *) A + pitch * BLOCK_SIZE * by + pitch * ty);
    float *rowB = b;
    float sum = 0.0f;
    for (int ia = aBegin; ia <= aEnd; ia += aStep) {
        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float bs[BLOCK_SIZE];
        
        as[ty][tx] = rowA[tx];
        bs[ty] = rowB[ty];
        __syncthreads();

        rowA = (float *)((char *) rowA + BLOCK_SIZE * sizeof(float));
        rowB += BLOCK_SIZE;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += as[ty][k] * bs[k];
        }
    }
    float *rowC = c + BLOCK_SIZE * by;
    float *rowG = g + BLOCK_SIZE * by;
    rowC[ty] = sum + rowG[ty];
}

// g = D * f
__global__ void multDiagMatAndVec(float *D, size_t pitchD, float *f, float *g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == j) {
        float *rowD = (float *)((char *) D + i * pitchD);
        g[i] = rowD[i] * f[i];
    }
}

__global__ void invertDiagMat(float *D, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i == j) {
        float *rowD = (float *)((char *) D + i * pitch);
        rowD[j] = 1 / rowD[j];
    }
}

__global__ void getDiagMat(float *A, size_t pitchA, float *D, size_t pitchD) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float *rowA = (float *)((char *) A + i * pitchA);
    float *rowD = (float *)((char *) D + i * pitchD);
    if (i == j) {
        rowD[j] = rowA[j];
    } else {
        rowD[j] = 0.0f;
    }
}

void assertVectors(float *a, float *b, int N) {
    for (int i = 0; i < N; i++) {
        assert(fabsf(a[i] - b[i]) < EPS);
    }
}

float normOfDifference(float *a, float *b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = a[i] - b[i];
        sum += val * val;
    }
    return sqrtf(sum);
}

void findSolution(float *ADev, size_t pitchA, float *f, float *x, int N) {
    float *fDev, *xDev, *xNextDev, *BDev, *DDev, *gDev, *DADiffDev;
    size_t pitchB, pitchD, pitchDADiff;
    cudaMallocPitch((void **) &BDev, &pitchB, N * sizeof(float), N);
    cudaMallocPitch((void **) &DDev, &pitchD, N * sizeof(float), N);
    cudaMallocPitch((void **) &DADiffDev, &pitchDADiff, N * sizeof(float), N);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    getDiagMat<<<numBlocks, threadsPerBlock>>>(ADev, pitchA, DDev, pitchD);
    matDiff<<<numBlocks, threadsPerBlock>>>(DDev, pitchD, ADev, pitchA, DADiffDev, pitchDADiff);
    invertDiagMat<<<numBlocks, threadsPerBlock>>>(DDev, pitchD);
    multDiagMatAndMat<<<numBlocks, threadsPerBlock>>>(DDev, pitchD, DADiffDev, pitchDADiff, BDev, pitchB);
    
    cudaFree(ADev);
    cudaFree(DADiffDev);

    cudaMalloc((void **) &fDev, N * sizeof(float));
    cudaMemcpy(fDev, f, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &gDev, N * sizeof(float));
    multDiagMatAndVec<<<numBlocks, threadsPerBlock>>>(DDev, pitchD, fDev, gDev);
    printVectorDev(gDev, N);

    cudaFree(DDev);    

    float *xNext = new float[N];
    cudaMalloc((void **) &xNextDev, N * sizeof(float));
    cudaMalloc((void **) &xDev, N * sizeof(float));
    cudaMemcpy(xDev, x, N * sizeof(float), cudaMemcpyHostToDevice);

    //float *BxDev;
    //cudaMalloc((void **) &BxDev, N * sizeof(float));
    int i = 0;
    float nnorm;
    while (true) {
        multMatVecPlusVec<<<numBlocks, threadsPerBlock>>>(BDev, pitchB, xDev, gDev, xNextDev, N);
        //xNextDev = B * xDev + g;
        //printf("----------\n");
        //printVectorDev(xDev, N);
        //printVectorDev(xNextDev, N);
        cudaMemcpy(x, xDev, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(xNext, xNextDev, N * sizeof(float), cudaMemcpyDeviceToHost);
        // copy xDev and xNextDev to host
        // find norm of difference
        // copy xNextDev to xDev
        nnorm = normOfDifference(xNext, x, N);
        //printf("Step: %d (%.5f)\n", i, nnorm);
        i++;
        if (nnorm < EPS) {
            break;
        }
        cudaMemcpy(xDev, xNextDev, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    // copy xNext to x
    printVector(xNext, N);
    cudaMemcpy(x, xNext, N * sizeof(float), cudaMemcpyHostToHost);
    //cudaFree(BxDev);
    cudaFree(xDev);
    cudaFree(xNextDev);
    cudaFree(BDev);
    delete xNext;
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
    float *f = new float[N];
    float *x = new float[N];
    for (int i = 0; i < N; i++) {
        f[i] = i + 1;
        x[i] = 0;
    }

    float *ADev;
    size_t pitchA;
    cudaMallocPitch((void **) &ADev, &pitchA, N * sizeof(float), N);
    cudaMemcpy2D(ADev, pitchA, A, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    printf("Threads per block: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Num blocks: (%d, %d)\n", numBlocks.x, numBlocks.y);

    fillRandom<<<numBlocks, threadsPerBlock>>>(ADev, N, pitchA);
    printMatrixDev(ADev, pitchA, N);
    findSolution(ADev, pitchA, f, x, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaFree(ADev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printMatrix(A, N);
    printVector(x, N);
    //mult(A, A, C2, N);
   // multMatVecClassic(A, f, x2, N);
   // assertVectors(x, x2, N);

    printf("Elapsed time: %.2f ms\n", gpuTime);

    /*printMatrix(A, N);
    printVector(x, N);
    printVector(x2, N);*/

    /*for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%3.0f ", A[i * N + j]);
        }
        printf("| %3.0f {%3.0f | %3.0f}\n", f[i], x2[i], x[i]);
    }*/
    /*printf("multiply:\n");
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
    }*/
    delete A;
    delete f;
    delete x;
    return 0;
}