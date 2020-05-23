#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#define EPS 0.0000001f
#define BLOCK_SIZE 4

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("[CUDA ERROR]: Error at %s:%d. ",__FILE__,__LINE__);\
    printf("Exit failure: %d\n", EXIT_FAILURE);\
    }} while(0)

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
    CUDA_CALL(cudaMemcpy2D(A, N * sizeof(float), ADev, pitch, N * sizeof(float),
        N, cudaMemcpyDeviceToHost));
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
    CUDA_CALL(cudaMemcpy(v, vDev, N * sizeof(float), cudaMemcpyDeviceToHost));
    printVector(v, N);
    delete v;
}

__global__ void fillRandom(float *A, int N, size_t pitch, float *rand) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float *row = (float *)((char *) A + i * pitch);
    //row[j] = i + j + 1;
    row[j] = rand[i * N + j];
    atomicAdd(&row[i], row[j] * 10.0f);
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

    float *rowA = (float *)((char *)A + pitchA * BLOCK_SIZE * by + pitchA * ty);
    float *rowB = (float *)((char *)B + sizeof(float) * BLOCK_SIZE * bx +
        pitchB * ty);
    
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
__global__ void multDiagMatAndMat(float *D, size_t pitchD, float *A,
                                  size_t pitchA, float *C, size_t pitchC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float *rowA = (float *)((char *) A + i * pitchA);
    float *rowD = (float *)((char *) D + i * pitchD);
    float *rowC = (float *)((char *) C + i * pitchC);
    rowC[j] = rowA[j] * rowD[i];
}

// A * b + g = c
__global__ void multMatVecPlusVec(float *A, size_t pitch, float *b, float *g,
    float *c, int N) {
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
    if (g != nullptr) {
        float *rowG = g + BLOCK_SIZE * by;
        rowC[ty] = sum + rowG[ty];
    } else {
        rowC[ty] = sum;
    }
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

float normOfDifference(float *a, float *b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = a[i] - b[i];
        sum += val * val;
    }
    return sqrtf(sum);
}

void checkSolution(float *ADev, size_t pitchA, float *xDev, float *f, int N,
    dim3 &threadsPerBlock, dim3 &numBlocks) {
    float *AXDev;
    float *AX = new float[N];
    CUDA_CALL(cudaMalloc((void **) &AXDev, N * sizeof(float)));
    multMatVecPlusVec<<<numBlocks, threadsPerBlock>>>(ADev, pitchA, xDev,
        nullptr, AXDev, N);
    CUDA_CALL(cudaMemcpy(AX, AXDev, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        float diff = fabsf(AX[i] - f[i]);
        if (diff > 0.001f) {
            printf("[ERROR] Invalid solution.\nActual 'A * x' [%.7f]:\n", diff);
            printVector(AX, N);
            printf("Expected 'f':\n");
            printVector(f, N);
            return;
        }
    }
    printf("[SUCCESS] Solution:\n");
    printVectorDev(xDev, N);
}

void findSolution(float *ADev, size_t pitchA, float *f, float *x, int N,
    dim3 &threadsPerBlock, dim3 &numBlocks) {
    float *fDev, *xDev, *xNextDev, *BDev, *DDev, *gDev, *DADiffDev;
    size_t pitchB, pitchD, pitchDADiff;
    CUDA_CALL(cudaMallocPitch((void **) &BDev, &pitchB, N * sizeof(float), N));
    CUDA_CALL(cudaMallocPitch((void **) &DDev, &pitchD, N * sizeof(float), N));
    CUDA_CALL(cudaMallocPitch((void **) &DADiffDev, &pitchDADiff,
        N * sizeof(float), N));

    getDiagMat<<<numBlocks, threadsPerBlock>>>(ADev, pitchA, DDev, pitchD);
    matDiff<<<numBlocks, threadsPerBlock>>>(DDev, pitchD, ADev, pitchA,
        DADiffDev,pitchDADiff);
    invertDiagMat<<<numBlocks, threadsPerBlock>>>(DDev, pitchD);
    multDiagMatAndMat<<<numBlocks, threadsPerBlock>>>(DDev, pitchD, DADiffDev,
        pitchDADiff, BDev, pitchB);
    
    CUDA_CALL(cudaFree(DADiffDev));

    CUDA_CALL(cudaMalloc((void **) &fDev, N * sizeof(float)));
    CUDA_CALL(cudaMemcpy(fDev, f, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &gDev, N * sizeof(float)));
    multDiagMatAndVec<<<numBlocks, threadsPerBlock>>>(DDev, pitchD, fDev, gDev);

    CUDA_CALL(cudaFree(DDev));    

    float *xNext = new float[N];
    CUDA_CALL(cudaMalloc((void **) &xNextDev, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &xDev, N * sizeof(float)));
    CUDA_CALL(cudaMemcpy(xDev, x, N * sizeof(float), cudaMemcpyHostToDevice));

    //printMatrixDev(ADev, pitchA, N);
    //printVectorDev(fDev, N);

    int steps = 0;
    float Norm = 0.0f;
    while (true) {
        multMatVecPlusVec<<<numBlocks, threadsPerBlock>>>(BDev, pitchB, xDev,
            gDev, xNextDev, N);
        CUDA_CALL(cudaMemcpy(x, xDev, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(xNext, xNextDev, N * sizeof(float),
            cudaMemcpyDeviceToHost));
        Norm = normOfDifference(xNext, x, N);
        steps++;
        if (Norm < EPS) {
            break;
        }
        //printf("step: %d (%.7f)\n", steps, Norm);
        if (isnan(Norm) || isinf(Norm)) {
            printf("ERROR: System solution diverges. Please check if your "
                   "matrix is diagonal dominant.\n");
            break;
        }
        CUDA_CALL(cudaMemcpy(xDev, xNextDev, N * sizeof(float),
            cudaMemcpyDeviceToDevice));
    }
    checkSolution(ADev, pitchA, xNextDev, f, N, threadsPerBlock, numBlocks);
    printf("Total step count: %d\n", steps);
    CUDA_CALL(cudaMemcpy(x, xNext, N * sizeof(float), cudaMemcpyHostToHost));
    CUDA_CALL(cudaFree(xDev));
    CUDA_CALL(cudaFree(xNextDev));
    CUDA_CALL(cudaFree(BDev));
    delete xNext;
}

int main(int argc, char **argv) {
    int N = BLOCK_SIZE;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
    printf("Given matrix size: %d\n", N);

    /// Round up to the next multiple of BLOCK_SIZE.
    if (N % BLOCK_SIZE != 0) {
        printf("[WARNING]: current matrix size is not a multiple of BLOCK_SIZE. "
               "It will be round to the next multiple of BLOCK_SIZE.\n");
        N = (N / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }
    
    printf("Current matrix size: %d\n", N);

    CUDA_CALL(cudaSetDevice(0));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start, 0));

    float *A = new float[N * N];
    float *f = new float[N];
    float *x = new float[N];
    for (int i = 0; i < N; i++) {
        f[i] = i + 1;
        x[i] = 0;
    }

    float *ADev;
    size_t pitchA;
    CUDA_CALL(cudaMallocPitch((void **) &ADev, &pitchA, N * sizeof(float), N));
    CUDA_CALL(cudaMemcpy2D(ADev, pitchA, A, N * sizeof(float),
        N * sizeof(float), N, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    printf("Threads per block: (%d, %d)\n",
           threadsPerBlock.x, threadsPerBlock.y);
    printf("Num blocks: (%d, %d)\n", numBlocks.x, numBlocks.y);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    unsigned long long clock = 1234ULL;
    curandSetPseudoRandomGeneratorSeed(gen, clock);
    printf("Seed: %llu\n", clock);
    float *randDev;
    CUDA_CALL(cudaMalloc((void **) &randDev, N * N * sizeof(float)));
    curandGenerateUniform(gen, randDev, N * N);

    fillRandom<<<numBlocks, threadsPerBlock>>>(ADev, N, pitchA, randDev);
    curandDestroyGenerator(gen);
    cudaFree(randDev);

    findSolution(ADev, pitchA, f, x, N, threadsPerBlock, numBlocks);

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&gpuTime, start, stop));

    CUDA_CALL(cudaFree(ADev));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    printf("Elapsed time: %.2f ms\n", gpuTime);

    delete A;
    delete f;
    delete x;
    return 0;
}