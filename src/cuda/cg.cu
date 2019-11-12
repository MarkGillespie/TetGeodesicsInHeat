#include "cg.cuh"

#define NTHREAD 1
#define NBLOCK  1

__global__ void computeAp(float *out, float *p, float *cotans, int* neighbors, int meshStride, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = index; i < n; i += stride){
        out[i] = 1e-12 * p[i];
        for (int iN = 0; iN < meshStride; ++iN) {
            int neighbor = neighbors[iN];
            double weight = cotans[iN];
            out[i] += weight * (p[i] - p[neighbor]);
        }
    }
}

__global__ void vector_sub(float *out, float *a, float *b, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    out[i] = a[i] - b[i];
  }
}

__global__ void vector_cpy(float *out, float *a, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    out[i] = a[i];
  }
}

__global__ void compute_alpha(float *out, float *r2, float *r, float *p, float *Ap, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    r2[0] = 0;
    float denom = 0;
    for (int i = 0; i < n; i += 1) {
      r2[0] += r[i] * r[i];
      denom += p[i] * Ap[i];
    }
    out[0] = r2[0] / denom;
  }
}

__global__ void update_x_r(float* x, float* r, float* alpha, float* p, float* Ap, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    x[i] += alpha[0] * p[i];
    r[i] -= alpha[0] * Ap[i];
  }
}

__global__ void compute_beta(float *out, float *r2, float *r, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    out[0] = 1/r2[0];
    r2[0] = 0;
    for (int i = 0; i < n; i += 1) {
      r2[0] += r[i] * r[i];
    }
    out[0] *= r2[0];
    r2[0] = 5;
  }
}

__global__ void set_r2(float *r2) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    r2[0] = 5;
  }
}

// Returns p = beta * p + r
__global__ void update_p(float* p, float *beta, float *r, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
      p[i] *= beta[0];
      p[i] += r[i];
    //p[i] = beta[0] * p[i] + r[i];
  }
}

std::vector<std::unordered_map<size_t, double>> edgeWeights(const TetMesh& mesh) {
    std::vector<std::unordered_map<size_t, double>> weights;
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        std::unordered_map<size_t, double> vWeights;
        weights.push_back(vWeights);
    }

    for (size_t iPE = 0; iPE < mesh.edges.size(); ++iPE) {
        PartialEdge pe = mesh.edges[iPE];
        size_t vSrc    = pe.src;
        size_t vDst    = pe.dst;

        double weight = mesh.partialEdgeCotanWeights[iPE];

        auto findSrc = weights[vSrc].find(vDst);
        if (findSrc == weights[vSrc].end()) {
            weights[vSrc][vDst] = weight;
        } else {
            findSrc->second += weight;
        }

        auto findDst = weights[vDst].find(vSrc);
        if (findDst == weights[vDst].end()) {
            weights[vDst][vSrc] = weight;
        } else {
            findDst->second += weight;
        }

    }

    return weights;
}

void cgSolve(Eigen::VectorXd& xOut, Eigen::VectorXd bVec, const TetMesh& mesh, double t) {
    float *x, *b, *r;
    float *d_x, *d_b, *d_p, *d_Ap, *d_r, *d_r2, *d_alpha, *d_beta;
    int N = bVec.size();

    // Allocate host memory
    x   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    r   = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        x[i] = 0.13f;
        b[i] = bVec[i];
        r[i] = b[i];
        if (abs(b[i]) > 0) {
            printf("\tBIG B: i = %d, \t b = %f\n", i, b[i]);
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x,     sizeof(float) * N);
    cudaMalloc((void**)&d_b,     sizeof(float) * N);
    cudaMalloc((void**)&d_Ap,    sizeof(float) * N);
    cudaMalloc((void**)&d_p,     sizeof(float) * N);
    cudaMalloc((void**)&d_r,     sizeof(float) * N);
    cudaMalloc((void**)&d_alpha, sizeof(float) * 1);
    cudaMalloc((void**)&d_beta,  sizeof(float) * 1);
    cudaMalloc((void**)&d_r2,    sizeof(float) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel

    bool done = false;
    int iter = 0;

    float *cotans;
    int *neighbors;
    int maxDegree = 0;
    std::vector<std::unordered_map<size_t, double>> weights = edgeWeights(mesh);
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        maxDegree = std::max(maxDegree, (int) weights[iV].size());
    }

    cotans    = (float*) malloc(sizeof(float) * maxDegree * N);
    neighbors = (int*)   malloc(sizeof(int)   * maxDegree * N);

    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        size_t neighborCount = 0;
        for (std::pair<size_t, double> elem : weights[iV]) {
            neighbors[iV * maxDegree + neighborCount] = elem.first;
            cotans[iV * maxDegree + neighborCount] = elem.second;
            ++neighborCount;
        }

        // Fill in the remaining slots with zeros
        for (size_t iN = neighborCount; iN < maxDegree; ++iN) {
            neighbors[iV * maxDegree + iN] = iV;
            cotans[iV * maxDegree + iN] = 0;
        }
    }
    printf("max degree: %d\n", maxDegree);

    //computeAp<<<NBLOCK,NTHREAD>>>(d_Ap, d_x, cotans, neighbors, maxDegree, N);
    //vector_sub<<<NBLOCK,NTHREAD>>>(d_r, d_b, d_Ap, N);
    //vector_cpy<<<NBLOCK,NTHREAD>>>(d_p, d_r, N);

    float *Ap = (float*)malloc(sizeof(float) * N);
    cudaMemcpy(r, d_r, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ap, d_Ap, sizeof(float) * N, cudaMemcpyDeviceToHost);
    printf("r[0] : %f \t r[1] : %f \t r[2] : %f\n", r[0], r[1], r[2]);
    printf("b[0] : %f \t b[1] : %f \t b[2] : %f\n", b[0], b[1], b[2]);
    printf("Ap[0] : %f \t Ap[1] : %f \t Ap[2] : %f\n", Ap[0], Ap[1], Ap[2]);
    printf("\n");
    while (!done) {
      for (int i = 0; i < 1; ++i) {
        // computeAp<<<NBLOCK,NTHREAD>>>(d_Ap, d_p, cotans, neighbors, maxDegree, N);
        // compute_alpha<<<NBLOCK, NTHREAD>>>(d_alpha, d_r2, d_r, d_p, d_Ap, N);
        // update_x_r<<<NBLOCK, NTHREAD>>>(d_x, d_r, d_alpha, d_p, d_Ap, N);
        compute_beta<<<NBLOCK, NTHREAD>>>(d_beta, d_r2, d_r, N);
        update_p<<<NBLOCK, NTHREAD>>>(d_p, d_beta, d_r, N);
      }
      compute_beta<<<NBLOCK, NTHREAD>>>(d_beta, d_r2, d_r, N);

      // Transfer data back to host memory
      cudaMemcpy(r, d_r, sizeof(float) * N, cudaMemcpyDeviceToHost);

      float *alpha = new float[1];
      //float *beta = new float[1];
      float *r2 = (float*)malloc(sizeof(float));
      //float *p = new float[N];
      //cudaMemcpy(p,     d_p,     sizeof(float) * N, cudaMemcpyDeviceToHost);
      //cudaMemcpy(x,     d_x,     sizeof(float) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(r2,    d_r2,    sizeof(float) * 1, cudaMemcpyDeviceToHost);
      cudaMemcpy(alpha, d_alpha, sizeof(float) * 1, cudaMemcpyDeviceToHost);
      //cudaMemcpy(beta,  d_beta,  sizeof(float) * 1, cudaMemcpyDeviceToHost);
      //float norm = 0;
      //for (int i = 0; i < N; i++) {
        //norm = fmax(norm, r[i] * r[i]);
      //}
      ++iter;
      printf("alpha: %f \t r^2  : %f\n", alpha[0], r2[0]);
      //printf("x[0] : %f \t x[1] : %f \t x[2] : %f\n", x[0], x[1], x[2]);
      //printf("p[0] : %f \t p[1] : %f \t p[2] : %f\n", p[0], p[1], p[2]);
      printf("r[0] : %f \t r[1] : %f \t r[2] : %f\n", r[0], r[1], r[2]);
      //printf("norm: %f\n", norm);
      //printf("\n");
      //fflush(stdout);
      //done = (norm < 1e-4) || (iter > 0);
      done = true;
    }

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; ++i) {
      float result = 1e-12;
      for (int iN = 0; iN < maxDegree; ++iN) {
          int neighbor = neighbors[iN];
          double weight = cotans[iN];
          result += weight * (x[i] - x[neighbor]);
      }
      if (abs(result - b[i]) > 1e-4) {
          printf("err: vertex %d result[%d] = %f, b[%d] = %f, x[%d] = %f\n", i, i, result, i, b[i], i, x[i]);
          printf("iter: %d\n", iter);
      }
      xOut[i] = x[i];
    }

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
    free(cotans);
    free(neighbors);

    return;
}
