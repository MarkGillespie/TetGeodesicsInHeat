#include "jacobi.cuh"

#define NTHREAD 256
#define NBLOCK  500

// Computes out = b - (tR)x where R is the matrix of off-diagonal entries of the Laplacian
__global__ void compute_b_minus_Rx(double *out, double *x, double* b, double *cotans, int* neighbors, int meshStride, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = b[i];
        for (int iN = 0; iN < meshStride; ++iN) {
            int neighbor = neighbors[i * meshStride + iN];
            double weight = cotans[i * meshStride + iN];
            out[i] += weight * x[neighbor];
        }
    }
}

// Computes x[i] = 2/3 * a[i] / b[i] + 1/3 * x[i]
__global__ void update_x(double *x, double *a, double *b, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    x[i] = 2. / 3. * a[i] / b[i] + 1. / 3. * x[i];
  }
}

// Computes residual i.e. (M + tL)x - b
__global__ void residual(double *out, double *x, double* b, double *cotans, int* neighbors, double* diag, int meshStride, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = diag[i] * x[i] - b[i];
        for (int iN = 0; iN < meshStride; ++iN) {
            int neighbor = neighbors[i * meshStride + iN];
            double weight = cotans[i * meshStride + iN];
            out[i] -= weight * x[neighbor];
        }
    }
}


// For each vertex, returns a map which maps the indices of the vertex's neighbors to the
// entries of the laplacian corresponding to those edges
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
        weights[vSrc][vDst] += weight;
        weights[vDst][vSrc] += weight;
    }

    return weights;
}

// If t < 0, solve Lx = b (realy we relax to (L + 1e-12)x = b to ensure our
// system is positive definite
// If t >= 0, solve (M + tL)x = b, where M is the mass matrix and L is the laplacian
int  jacobiSolve(Eigen::VectorXd& xOut, Eigen::VectorXd bVec, const TetMesh& mesh, double tol, double t) {
    double *x, *b, *r, *cotans, *diag;
    double *d_x, *d_b, *d_b_minus_Rx, *d_r, *d_cotans, *d_diag;
    int* neighbors, *d_neighbors;
    int N = bVec.size();

    int maxDegree = 0;
    std::vector<std::unordered_map<size_t, double>> weights = edgeWeights(mesh);
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        maxDegree = std::max(maxDegree, (int) weights[iV].size());
    }

    // Allocate host memory
    x         = (double*) malloc(sizeof(double) * N);
    b         = (double*) malloc(sizeof(double) * N);
    r         = (double*) malloc(sizeof(double) * N);
    diag      = (double*) malloc(sizeof(double) * N);
    cotans    = (double*) malloc(sizeof(double) * N * maxDegree);
    neighbors = (int*   ) malloc(sizeof(int   ) * N * maxDegree);

    // Initialize host arrays
    for(size_t iV = 0; iV < N; iV++){
        x[iV] = 0.0f;
        b[iV] = bVec[iV];
        diag[iV] = (t < 0)?1e-5:mesh.vertexDualVolumes[iV];
        if (iV < 5)
            printf("%lu %f\n", iV, diag[iV]);
    }
    if (t < 0) t = 1;

    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        size_t neighborCount = 0;
        double totalWeight = 0;
        for (std::pair<size_t, double> elem : weights[iV]) {
            neighbors[iV * maxDegree + neighborCount] = elem.first;
            cotans[iV * maxDegree + neighborCount] = t * elem.second;
            totalWeight += t * elem.second;
            ++neighborCount;
        }
        diag[iV] += t * totalWeight;
        if (iV < 5)
            printf("%lu %f\n", iV, diag[iV]);

        // Fill in the remaining slots with zeros
        for (size_t iN = neighborCount; iN < maxDegree; ++iN) {
            neighbors[iV * maxDegree + iN] = iV;
            cotans[iV * maxDegree + iN] = 0;
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x,          sizeof(double) * N);
    cudaMalloc((void**)&d_b,          sizeof(double) * N);
    cudaMalloc((void**)&d_b_minus_Rx, sizeof(double) * N);
    cudaMalloc((void**)&d_r,          sizeof(double) * N);
    cudaMalloc((void**)&d_diag,       sizeof(double) * N);
    cudaMalloc((void**)&d_neighbors,  sizeof(int   ) * N * maxDegree);
    cudaMalloc((void**)&d_cotans,     sizeof(double) * N * maxDegree);

    // Transfer data from host to device memory
    cudaMemcpy(d_x,         x,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,         b,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_diag,      diag,      sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors, sizeof(int   ) * N * maxDegree, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cotans,    cotans,    sizeof(double) * N * maxDegree, cudaMemcpyHostToDevice);

    bool done = false;
    int iter = 0;

    int substeps = 40;
    while (!done) {
      for (int i = 0; i < substeps; ++i) {
        compute_b_minus_Rx<<<NBLOCK, NTHREAD>>>(d_b_minus_Rx, d_x, d_b, d_cotans, d_neighbors,
                                                maxDegree, N);
        update_x<<<NBLOCK, NTHREAD>>>(d_x, d_b_minus_Rx, d_diag, N);
      }

      residual<<<NBLOCK, NTHREAD>>>(d_r, d_x, d_b, d_cotans, d_neighbors, d_diag, maxDegree, N);

      // Transfer data back to host memory
      cudaMemcpy(r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost);
      double norm = 0;
      for (int i = 0; i < N; i++) {
        norm = fmax(norm, fabs(r[i]));
      }
      ++iter;
      printf("%d: residual: %f\n", iter, norm);
      done = (norm < tol) || (iter > 5);
    }

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; ++i) {
      double result = diag[i] * x[i];
      for (int iN = 0; iN < maxDegree; ++iN) {
          int neighbor  = neighbors[i * maxDegree + iN];
          double weight =    cotans[i * maxDegree + iN];
          result += t * weight * (x[i] - x[neighbor]);
      }
      if (abs(result - b[i]) > tol) {
          //printf("err: vertex %d result[%d] = %f, b[%d] = %f, x[%d] = %f, err=%.10e, iter=%d\n", i, i, result, i, b[i], i, x[i], result-b[i], iter);
      }
      xOut[i] = x[i];
    }

    // Deallocate device memory
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_b_minus_Rx);
    cudaFree(d_r);
    cudaFree(d_diag);
    cudaFree(d_neighbors);
    cudaFree(d_cotans);

    // Deallocate host memory
    free(x);
    free(b);
    free(r);
    free(cotans);
    free(neighbors);
    free(diag);

    return iter * substeps;
}
