#include "cg.cuh"

#define N 100000
#define NTHREAD 256
#define NBLOCK 5000

__global__ void computeAp(float* out, float* p, int n) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = p[i] + 1 / 3. * (p[(i + n - 1) % n] + p[(i + 1) % n]);
    }
}

__global__ void vector_sub(float* out, float* a, float* b, int n) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] += a[i] - b[i];
    }
}

__global__ void vector_cpy(float* out, float* a, int n) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] += a[i];
    }
}

__global__ void compute_alpha(float* out, float* r2, float* r, float* p,
                              float* Ap, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) {
        r2[0]       = 0;
        float denom = 0;
        for (int i = 0; i < n; i += 1) {
            r2[0] += r[i] * r[i];
            denom += p[i] * Ap[i];
        }
        out[0] = r2[0] / denom;
    }
}

__global__ void update_x_r(float* x, float* r, float* alpha, float* p,
                           float* Ap, int n) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        x[i] += alpha[0] * p[i];
        r[i] -= alpha[0] * Ap[i];
    }
}

// Returns out = s * out + b
__global__ void update_p(float* p, float* alpha, float* r, int n) {
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        p[i] = alpha[0] * p[i] + r[i];
    }
}

__global__ void compute_beta(float* out, float* r2, float* r, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) {
        out[0] = 1 / r2[0];
        r2[0]  = 0;
        for (int i = 0; i < n; i += 1) {
            r2[0] += r[i] * r[i];
        }
        out[0] *= r2[0];
    }
}

namespace CompArch {
std::vector<double> solveCG(const TetMesh& mesh, std::vector<double> rhs) {
    std::vector<double> solution;
    return solution;
}
} // namespace CompArch

int main() {
    float *x, *b, *r;
    float *d_x, *d_b, *d_p, *d_Ap, *d_r, *d_r2, *d_alpha, *d_beta;

    // Allocate host memory
    x = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    r = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        b[i] = i % 4;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_Ap, sizeof(float) * N);
    cudaMalloc((void**)&d_p, sizeof(float) * N);
    cudaMalloc((void**)&d_r, sizeof(float) * N);
    cudaMalloc((void**)&d_alpha, sizeof(float) * 1);
    cudaMalloc((void**)&d_beta, sizeof(float) * 1);
    cudaMalloc((void**)&d_r2, sizeof(float) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel

    bool done = false;
    int iter  = 0;

    computeAp<<<NBLOCK, NTHREAD>>>(d_Ap, d_x, N);
    vector_sub<<<NBLOCK, NTHREAD>>>(d_r, d_b, d_Ap, N);
    vector_cpy<<<NBLOCK, NTHREAD>>>(d_p, d_r, N);
    while (!done) {
        for (int i = 0; i < 3; ++i) {
            computeAp<<<NBLOCK, NTHREAD>>>(d_Ap, d_p, N);
            compute_alpha<<<NBLOCK, NTHREAD>>>(d_alpha, d_r2, d_r, d_p, d_Ap,
                                               N);
            update_x_r<<<NBLOCK, NTHREAD>>>(d_x, d_r, d_alpha, d_p, d_Ap, N);
            compute_beta<<<NBLOCK, NTHREAD>>>(d_beta, d_r2, d_r, N);
            update_p<<<NBLOCK, NTHREAD>>>(d_p, d_beta, d_r, N);
        }

        // Transfer data back to host memory
        cudaMemcpy(r, d_r, sizeof(float) * N, cudaMemcpyDeviceToHost);

        float* alpha = new float[1];
        float* beta  = new float[1];
        float* p     = new float[N];
        cudaMemcpy(p, d_p, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(x, d_x, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(alpha, d_alpha, sizeof(float) * 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(beta, d_beta, sizeof(float) * 1, cudaMemcpyDeviceToHost);
        float norm = 0;
        for (int i = 0; i < N; i++) {
            norm += r[i] * r[i];
        }
        ++iter;
        printf("r : %f, %f, %f, %f\n", r[0], r[1], r[2], r[3]);
        printf("r2: %f\n", norm);
        printf("p : %f, %f, %f, %f\n", p[0], p[1], p[2], p[3]);
        printf("x : %f, %f, %f, %f\n", x[0], x[1], x[2], x[3]);
        printf("α : %f\n", alpha[0]);
        printf("β : %f\n", beta[0]);
        done = (norm < 1e-4) || (iter > 100);
        printf("\n");

        // done = true;
    }

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; ++i) {
        float result = x[i] + 1 / 3. * (x[(i + N - 1) % N] + x[(i + 1) % N]);
        if (abs(result - b[i]) > 1e-4) {
            printf("err: %d result[%d] = %f, b[%d] = %f\n", i, i, result, i,
                   b[i]);
        }
    }
    printf("x[0] = %f\n", x[0]);

    // Deallocate device memory
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_Ap);
    cudaFree(d_p);
    cudaFree(d_r);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_r2);

    // Deallocate host memory
    free(x);
    free(b);
}
