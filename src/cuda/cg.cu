#include "cg.cuh"

#define NTHREAD 256
#define NBLOCK  500

// Computes out = (M + tL)p
__global__ void computeAp(double *out, double *p, double *cotans, int* neighbors, double* m, double t, int meshStride, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = index; i < n; i += stride){
        out[i] = m[i] * p[i];
        for (int iN = 0; iN < meshStride; ++iN) {
            int neighbor = neighbors[i * meshStride + iN];
            double weight = cotans[i * meshStride + iN];
            out[i] += t * weight * (p[i] - p[neighbor]);
        }
    }
}

// Computes out = a-b
__global__ void vector_sub(double *out, double *a, double *b, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    out[i] = a[i] - b[i];
  }
}

// Copies a into out
__global__ void vector_cpy(double *out, double *a, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    out[i] = a[i];
  }
}

// Computes alpha = dot(r, r) / dot(p, Ap). Returns dot(r, r) as r2
__global__ void compute_alpha(double *out, double *r2, double *r, double *p, double *Ap, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    r2[0] = 0;
    double denom = 0;
    for (int i = 0; i < n; i += 1) {
      r2[0] += r[i] * r[i];
      denom += p[i] * Ap[i];
    }
    out[0] = r2[0] / denom;
  }
}

// x += alpha p
// r -= alpha Ap
__global__ void update_x_r(double* x, double* r, double* alpha, double* p, double* Ap, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    x[i] += alpha[0] * p[i];
    r[i] -= alpha[0] * Ap[i];
  }
}

// Computes beta = dot(r, r) / r2
// where r2 is the old value of dot(r, r).
// Also updates r2 to be the new value of dot(r, r)
__global__ void compute_beta(double *out, double *r2, double *r, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    out[0] = 1/r2[0];
    r2[0] = 0;
    for (int i = 0; i < n; i += 1) {
      r2[0] += r[i] * r[i];
    }
    out[0] *= r2[0];
  }
}

// Computes p = beta * p + r
__global__ void update_p(double* p, double *beta, double *r, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int i = index; i < n; i += stride){
    p[i] = beta[0] * p[i] + r[i];
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
void cgSolve(Eigen::VectorXd& xOut, Eigen::VectorXd bVec, const TetMesh& mesh, double tol, double t) {
    double *x, *b, *r, *cotans, *m;
    double *d_x, *d_b, *d_p, *d_Ap, *d_r, *d_r2, *d_alpha, *d_beta, *d_cotans, *d_m;
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
    m         = (double*) malloc(sizeof(double) * N);
    cotans    = (double*) malloc(sizeof(double) * N * maxDegree);
    neighbors = (int*  ) malloc(sizeof(int  ) * N * maxDegree);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        x[i] = 0.0f;
        b[i] = bVec[i];
        r[i] = b[i];
        m[i] = (t < 0)?1e-12:mesh.vertexDualVolumes[i];
        //m[i] = (t < 0)?1:mesh.vertexDualVolumes[i];
    }
    if (t < 0) t = 1;

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
    Eigen::SparseMatrix<double> L    = mesh.weakLaplacian();
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        for (size_t iN = 0; iN < maxDegree; ++iN) {
            size_t jV = neighbors[iV * maxDegree + iN];
            if (jV == iV) continue;

            double mat = L.coeffRef(iV, jV);
            double arr = cotans[iV * maxDegree + iN];
            if (abs(mat + arr) >= 1e-7) {
                printf("ERROR: matrix is %f\tarray is %f\terror is %.10e\n",
                        mat, arr, abs(mat + arr));
            }
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x,         sizeof(double) * N);
    cudaMalloc((void**)&d_b,         sizeof(double) * N);
    cudaMalloc((void**)&d_Ap,        sizeof(double) * N);
    cudaMalloc((void**)&d_p,         sizeof(double) * N);
    cudaMalloc((void**)&d_r,         sizeof(double) * N);
    cudaMalloc((void**)&d_m,         sizeof(double) * N);
    cudaMalloc((void**)&d_neighbors, sizeof(int  ) * N * maxDegree);
    cudaMalloc((void**)&d_cotans,    sizeof(double) * N * maxDegree);
    cudaMalloc((void**)&d_alpha,     sizeof(double) * 1);
    cudaMalloc((void**)&d_beta,      sizeof(double) * 1);
    cudaMalloc((void**)&d_r2,        sizeof(double) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_x,         x,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,         r,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,         b,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,         m,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors, sizeof(int  ) * N * maxDegree, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cotans,    cotans,    sizeof(double) * N * maxDegree, cudaMemcpyHostToDevice);

    bool done = false;
    int iter = 0;

    computeAp<<<NBLOCK,NTHREAD>>>(d_Ap, d_x, d_cotans, d_neighbors, d_m, t, maxDegree, N);
    vector_sub<<<NBLOCK,NTHREAD>>>(d_r, d_b, d_Ap, N);
    vector_cpy<<<NBLOCK,NTHREAD>>>(d_p, d_r, N);
    {
      cudaMemcpy(r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost);

      double *alpha = new double[1];
      //double *beta = new double[1];
      double *r2 = (double*)malloc(sizeof(double));
      double *p = new double[N];
      double *Ap = new double[N];
      cudaMemcpy(p,     d_p,     sizeof(double) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(Ap,    d_Ap,    sizeof(double) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(x,     d_x,     sizeof(double) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(b,     d_b,     sizeof(double) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(r2,    d_r2,    sizeof(double) * 1, cudaMemcpyDeviceToHost);
      cudaMemcpy(alpha, d_alpha, sizeof(double) * 1, cudaMemcpyDeviceToHost);
      //cudaMemcpy(beta,  d_beta,  sizeof(double) * 1, cudaMemcpyDeviceToHost);
      double norm = 0;
      for (int i = 0; i < N; i++) {
        norm = fmax(norm, fabs(r[i]));
      }
      ++iter;
      //printf("r2  : %f\n", r2[0]);
      printf("alpha: %f \t r2  : %f\n", alpha[0], r2[0]);
      printf("x[0] : %.10e \t x[1] : %.10e \t x[2] : %.10e\n", x[0], x[1], x[2]);
      printf("b[0] : %.10e \t b[1] : %.10e \t b[2] : %.10e\n", b[0], b[1], b[2]);
      printf("p[0] : %.10e \t p[1] : %.10e \t p[2] : %.10e\n", p[0], p[1], p[2]);
      printf("Ap[0] : %.10e \t Ap[1] : %.10e \t Ap[2] : %.10e\n", Ap[0], Ap[1], Ap[2]);
      printf("r[0] : %.10e \t r[1] : %.10e \t r[2] : %.10e\n", r[0], r[1], r[2]);
    }
    while (!done) {
      for (int i = 0; i < 40; ++i) {
         computeAp<<<NBLOCK,NTHREAD>>>(d_Ap, d_p, d_cotans, d_neighbors, d_m, t, maxDegree, N);
         compute_alpha<<<NBLOCK, NTHREAD>>>(d_alpha, d_r2, d_r, d_p, d_Ap, N);
         update_x_r<<<NBLOCK, NTHREAD>>>(d_x, d_r, d_alpha, d_p, d_Ap, N);
         compute_beta<<<NBLOCK, NTHREAD>>>(d_beta, d_r2, d_r, N);
         update_p<<<NBLOCK, NTHREAD>>>(d_p, d_beta, d_r, N);
      }

      // Transfer data back to host memory
      cudaMemcpy(r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost);

      double *alpha = new double[1];
      //double *beta = new double[1];
      double *r2 = (double*)malloc(sizeof(double));
      double *p = new double[N];
      cudaMemcpy(p,     d_p,     sizeof(double) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(x,     d_x,     sizeof(double) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(r2,    d_r2,    sizeof(double) * 1, cudaMemcpyDeviceToHost);
      cudaMemcpy(alpha, d_alpha, sizeof(double) * 1, cudaMemcpyDeviceToHost);
      //cudaMemcpy(beta,  d_beta,  sizeof(double) * 1, cudaMemcpyDeviceToHost);
      double norm = 0;
      for (int i = 0; i < N; i++) {
        norm = fmax(norm, fabs(r[i]));
      }
      ++iter;
      //printf("r2  : %f\n", r2[0]);
      printf("alpha: %f \t r2  : %f\n", alpha[0], r2[0]);
      printf("x[0] : %.10e \t x[1] : %.10e \t x[2] : %.10e\n", x[0], x[1], x[2]);
      printf("p[0] : %.10e \t p[1] : %.10e \t p[2] : %.10e\n", p[0], p[1], p[2]);
      printf("r[0] : %.10e \t r[1] : %.10e \t r[2] : %.10e\n", r[0], r[1], r[2]);
      //printf("norm: %f\n", norm);
      //printf("\n");
      //fflush(stdout);
      done = (norm < tol) || (iter > 100);
      //done = true;
    }

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; ++i) {
      double result = m[i] * x[i];
      for (int iN = 0; iN < maxDegree; ++iN) {
          int neighbor  = neighbors[i * maxDegree + iN];
          double weight =    cotans[i * maxDegree + iN];
          result += t * weight * (x[i] - x[neighbor]);
      }
      if (abs(result - b[i]) > tol) {
          printf("err: vertex %d result[%d] = %f, b[%d] = %f, x[%d] = %f, err=%.10e\n", i, i, result, i, b[i], i, x[i], result-b[i]);
          //printf("iter: %d\n", iter);
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
