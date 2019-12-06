#include "cg.cuh"

#define NTHREAD 256
#define NBLOCK 500

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

// Computes out = (M + tL)p
__global__ void computeApCSR(double *out, double *p, double *cotans, int* neighbors, double* m, int* end, double t, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = m[i] * p[i];
        for (int iN = end[i-1]; iN < end[i]; ++iN) {
            int neighbor = neighbors[iN];
            double weight = cotans[iN];
            out[i] += t * weight * (p[i] - p[neighbor]);
        }
    }
}

// Computes out = (M + tL)p
// TODO: make sure there are enough blocks
extern __shared__ float localArray[];
__global__ void computeApClusteredCSR(double *out,
                                      double *global_p,
                                      double *global_cotans,
                                      int* global_neighbors,
                                      int* global_vertex_end,
                                      int* global_cluster_neighbors,
                                      int* global_cluster_neighbors_end,
                                      int* cluster_end,
                                      double* m,
                                      double t,
                                      int n) {

    //int stride = gridDim.x * blockDim.x;

    int cluster_starting_vert = cluster_end[2 * blockIdx.x + 0];
    int cluster_starting_edge = cluster_end[2 * blockIdx.x + 1];
    int cluster_ending_vert   = cluster_end[2 * blockIdx.x + 2];
    int cluster_ending_edge   = cluster_end[2 * blockIdx.x + 3];

    int verts_in_cluster = cluster_ending_vert - cluster_starting_vert;
    int edges_in_cluster = cluster_ending_edge - cluster_starting_edge;
    int neighbors_of_cluster = global_cluster_neighbors_end[blockIdx.x + 1] - global_cluster_neighbors_end[blockIdx.x];

    double* cotans    = (double*)  localArray;
    int*    neighbors = (int*)    &localArray[8 * edges_in_cluster];
    int*    end       = (int*)    &localArray[8 * edges_in_cluster + 4 * edges_in_cluster];
    double* p         = (double*) &localArray[8 * edges_in_cluster + 4 * edges_in_cluster + 4 * verts_in_cluster];

    if (threadIdx.x == 0) {
        for (int iE = 0; iE < edges_in_cluster; ++iE) {
            cotans[iE]    = global_cotans[cluster_starting_edge + iE];
            neighbors[iE] = global_neighbors[cluster_starting_edge + iE];
        }

        for (int iV = 0; iV < verts_in_cluster; ++iV) {
            p[iV] = global_p[cluster_starting_vert + iV];
        }

        for (int iN = 0; iN < neighbors_of_cluster; ++iN) {
            p[verts_in_cluster + iN] = global_cluster_neighbors[global_cluster_neighbors_end[blockIdx.x] + iN];
        }
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = m[i] * p[i];
    for (int iN = end[i-1]; iN < end[i]; ++iN) {
        int neighbor = neighbors[iN];
        double weight = cotans[iN];
        out[i] += t * weight * (p[i] - p[neighbor]);
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

// Computes out = num / denom
__global__ void div(double *out, double *num, double *denom) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
      *out = *num / *denom;
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
int  cgSolve(Eigen::VectorXd& xOut, Eigen::VectorXd bVec, const TetMesh& mesh, double tol, double t, bool verbose) {
    double *x, *b, *r, *cotans, *m;
    double *d_x, *d_b, *d_p, *d_Ap, *d_r, *d_old_r2, *d_new_r2,  *d_pAp, *d_alpha, *d_beta, *d_cotans, *d_m;
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

    //printf("max degree: %d\n", maxDegree);
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
    cudaMalloc((void**)&d_neighbors, sizeof(int   ) * N * maxDegree);
    cudaMalloc((void**)&d_cotans,    sizeof(double) * N * maxDegree);
    cudaMalloc((void**)&d_alpha,     sizeof(double) * 1);
    cudaMalloc((void**)&d_beta,      sizeof(double) * 1);
    cudaMalloc((void**)&d_old_r2,    sizeof(double) * 1);
    cudaMalloc((void**)&d_new_r2,    sizeof(double) * 1);
    cudaMalloc((void**)&d_pAp,       sizeof(double) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_x,         x,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,         r,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,         b,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,         m,         sizeof(double) * N,             cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors, sizeof(int   ) * N * maxDegree, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cotans,    cotans,    sizeof(double) * N * maxDegree, cudaMemcpyHostToDevice);

    bool done = false;
    int iter = 0;

    // https://stackoverflow.com/questions/12400477/retaining-dot-product-on-gpgpu-using-cublas-routine/12401838#12401838
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE); 

    computeAp<<<NBLOCK,NTHREAD>>>(d_Ap, d_x, d_cotans, d_neighbors, d_m, t, maxDegree, N);
    vector_sub<<<NBLOCK,NTHREAD>>>(d_r, d_b, d_Ap, N);
    vector_cpy<<<NBLOCK,NTHREAD>>>(d_p, d_r, N);
    cublasDdot(handle, N, d_r, 1, d_r, 1, d_old_r2);

    int substeps = 40;
    double norm = 0;
    while (!done) {
      for (int i = 0; i < substeps; ++i) {
         computeAp<<<NBLOCK,NTHREAD>>>(d_Ap, d_p, d_cotans, d_neighbors, d_m, t, maxDegree, N);
         cublasDdot(handle, N, d_p, 1, d_Ap, 1, d_pAp);
         div<<<NBLOCK, NTHREAD>>>(d_alpha, d_old_r2, d_pAp);
         update_x_r<<<NBLOCK, NTHREAD>>>(d_x, d_r, d_alpha, d_p, d_Ap, N);
         cublasDdot(handle, N, d_r, 1, d_r, 1, d_new_r2);
         div<<<NBLOCK, NTHREAD>>>(d_beta, d_new_r2, d_old_r2);
         update_p<<<NBLOCK, NTHREAD>>>(d_p, d_beta, d_r, N);
         vector_cpy<<<NBLOCK,NTHREAD>>>(d_old_r2, d_new_r2, 1);
      }

      // Transfer data back to host memory
      cudaMemcpy(r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost);
      norm = 0;
      for (int i = 0; i < N; i++) {
        norm = fmax(norm, fabs(r[i]));
      }
      ++iter;
      if (verbose) printf("%d: residual: %f\n", iter, norm);
      done = (norm < tol) || (iter > 300);
    }
    cublasDestroy(handle);

    if (norm >= tol)
        printf("timed out :'(");

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
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
    cudaFree(d_old_r2);
    cudaFree(d_new_r2);
    cudaFree(d_neighbors);
    cudaFree(d_cotans);

    // Deallocate host memory
    free(x);
    free(b);
    free(cotans);
    free(neighbors);

    return iter * substeps;
}

// If t < 0, solve Lx = b (realy we relax to (L + 1e-12)x = b to ensure our
// system is positive definite
// If t >= 0, solve (M + tL)x = b, where M is the mass matrix and L is the laplacian
// Stores matrix in CSR format
int cgSolveCSR(Eigen::VectorXd& xOut, Eigen::VectorXd bVec, const TetMesh& mesh, double tol, double t, bool verbose) {
    double *x, *b, *r, *cotans, *m;
    double *d_x, *d_b, *d_p, *d_Ap, *d_r, *d_old_r2, *d_new_r2,  *d_pAp, *d_alpha, *d_beta, *d_cotans, *d_m;
    int* neighbors, *d_neighbors, *end, *d_end;
    int N = bVec.size();

    std::vector<std::unordered_map<size_t, double>> weights = edgeWeights(mesh);
    int nEdges = 0;
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        nEdges += weights[iV].size();
    }

    // Allocate host memory
    x         = (double*) malloc(sizeof(double) * N);
    b         = (double*) malloc(sizeof(double) * N);
    r         = (double*) malloc(sizeof(double) * N);
    m         = (double*) malloc(sizeof(double) * N);
    cotans    = (double*) malloc(sizeof(double) * nEdges);
    neighbors = (int*   ) malloc(sizeof(int   ) * nEdges);
    end = (int*   ) malloc(sizeof(int   ) * 2 * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        x[i] = 0.0f;
        b[i] = bVec[i];
        r[i] = b[i];
        m[i] = (t < 0)?1e-12:mesh.vertexDualVolumes[i];
    }
    if (t < 0) t = 1;

    int pos = 0;
    end[0] = 0;
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        for (std::pair<size_t, double> elem : weights[iV]) {
            neighbors[pos] = elem.first;
            cotans[pos] = elem.second;
            pos += 1;
        }
        end[iV] = pos;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x,         sizeof(double) * N);
    cudaMalloc((void**)&d_b,         sizeof(double) * N);
    cudaMalloc((void**)&d_Ap,        sizeof(double) * N);
    cudaMalloc((void**)&d_p,         sizeof(double) * N);
    cudaMalloc((void**)&d_r,         sizeof(double) * N);
    cudaMalloc((void**)&d_m,         sizeof(double) * N);
    cudaMalloc((void**)&d_neighbors, sizeof(int   ) * nEdges);
    cudaMalloc((void**)&d_cotans,    sizeof(double) * nEdges);
    cudaMalloc((void**)&d_end,       sizeof(int   ) * (N+1));
    cudaMalloc((void**)&d_alpha,     sizeof(double) * 1);
    cudaMalloc((void**)&d_beta,      sizeof(double) * 1);
    cudaMalloc((void**)&d_old_r2,    sizeof(double) * 1);
    cudaMalloc((void**)&d_new_r2,    sizeof(double) * 1);
    cudaMalloc((void**)&d_pAp,       sizeof(double) * 1);

    // Transfer data from host to device memory
    cudaMemcpy(d_x,         x,         sizeof(double) * N,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,         r,         sizeof(double) * N,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,         b,         sizeof(double) * N,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,         m,         sizeof(double) * N,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors, sizeof(int   ) * nEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_end,       end,       sizeof(int   ) * (N+1),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_cotans,    cotans,    sizeof(double) * nEdges, cudaMemcpyHostToDevice);

    bool done = false;
    int iter = 0;

    // https://stackoverflow.com/questions/12400477/retaining-dot-product-on-gpgpu-using-cublas-routine/12401838#12401838
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    computeApCSR<<<NBLOCK,NTHREAD>>>(d_Ap, d_x, d_cotans, d_neighbors, d_m, d_end, t, N);
    vector_sub<<<NBLOCK,NTHREAD>>>(d_r, d_b, d_Ap, N);
    vector_cpy<<<NBLOCK,NTHREAD>>>(d_p, d_r, N);
    cublasDdot(handle, N, d_r, 1, d_r, 1, d_old_r2);

    int substeps = 40;
    double norm = 0;
    while (!done) {
      for (int i = 0; i < substeps; ++i) {
        computeApCSR<<<NBLOCK,NTHREAD>>>(d_Ap, d_p, d_cotans, d_neighbors, d_m, d_end, t, N);
        cublasDdot(handle, N, d_p, 1, d_Ap, 1, d_pAp);
        div<<<NBLOCK, NTHREAD>>>(d_alpha, d_old_r2, d_pAp);
        update_x_r<<<NBLOCK, NTHREAD>>>(d_x, d_r, d_alpha, d_p, d_Ap, N);
        cublasDdot(handle, N, d_r, 1, d_r, 1, d_new_r2);
        div<<<NBLOCK, NTHREAD>>>(d_beta, d_new_r2, d_old_r2);
        update_p<<<NBLOCK, NTHREAD>>>(d_p, d_beta, d_r, N);
        vector_cpy<<<NBLOCK,NTHREAD>>>(d_old_r2, d_new_r2, 1);
      }

      // Transfer data back to host memory
      cudaMemcpy(r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost);
      norm = 0;
      for (int i = 0; i < N; i++) {
        norm = fmax(norm, fabs(r[i]));
      }
      ++iter;
      if (verbose) printf("%d: residual: %f\n", iter, norm);
      done = (norm < tol) || (iter > 300);
    }
    if (norm >= tol)
        printf("timed out :'(");
    cublasDestroy(handle);

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
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
    cudaFree(d_old_r2);
    cudaFree(d_new_r2);
    cudaFree(d_neighbors);
    cudaFree(d_end);
    cudaFree(d_cotans);

    // Deallocate host memory
    free(x);
    free(b);
    free(cotans);
    free(neighbors);

    return iter * substeps;
}

// If t < 0, solve Lx = b (realy we relax to (L + 1e-12)x = b to ensure our
// system is positive definite
// If t >= 0, solve (M + tL)x = b, where M is the mass matrix and L is the laplacian
// Stores matrix in CSR format
int cgSolveClusteredCSR(Eigen::VectorXd& xOut, Eigen::VectorXd& bVec, const TetMesh& mesh, double tol, double t, bool verbose) {
    double *x, *b, *r, *cotans, *m;
    double *d_x, *d_b, *d_p, *d_Ap, *d_r, *d_old_r2, *d_new_r2,  *d_pAp, *d_alpha, *d_beta, *d_cotans, *d_m;
    int* neighbors, *d_neighbors, *vertex_end, *d_vertex_end, *cluster_end, *d_cluster_end;
    int  *cluster_neighbors, *d_cluster_neighbors, *cluster_neighbors_end, *d_cluster_neighbors_end;
    int N = bVec.size();

    printf("Started again\n");

    std::vector<std::unordered_map<size_t, double>> weights = edgeWeights(mesh);
    int nEdges = 0;
    for (size_t iV = 0; iV < mesh.vertices.size(); ++iV) {
        nEdges += weights[iV].size();
    }

    size_t clusterSize = 256;
    std::vector<std::vector<size_t>> clusters, clusterNeighbors;
    std::tie(clusters, clusterNeighbors) = clusterAndNeighbors(mesh, clusterSize);
    size_t nClusters = clusters.size();
    std::vector<size_t> vertexReordering(N, 0);        // vertexReordering[position in big array] = iV
    //std::vector<size_t> inverseVertexReordering(N, 0); // inverseVertexReordering[iV] = position in big array
    std::vector<std::unordered_map<size_t, size_t>> localInverseReordering;

    size_t count = 0;
    size_t nNeighbors = 0;
    for (size_t iC = 0; iC < nClusters; ++iC) {
        std::vector<size_t> cluster = clusters[iC];
        std::unordered_map<size_t, size_t> clusterInverseMap;
        size_t localCount = 0;
        for (size_t iV : cluster) {
            vertexReordering[count] = iV;
            clusterInverseMap[iV] = localCount;
            //inverseVertexReordering[iV] = count;
            count++;
            localCount++;
        }
        for (size_t iN : clusterNeighbors[iC]) {
            clusterInverseMap[iN] = localCount;
            localCount++;
            nNeighbors++;
        }
        localInverseReordering.push_back(clusterInverseMap);
    }

    // Allocate host memory
    x           = (double*) malloc(sizeof(double) * N);
    b           = (double*) malloc(sizeof(double) * N);
    r           = (double*) malloc(sizeof(double) * N);
    m           = (double*) malloc(sizeof(double) * N);
    cotans      = (double*) malloc(sizeof(double) * nEdges);
    neighbors   = (int*   ) malloc(sizeof(int   ) * nEdges);
    vertex_end  = (int*   ) malloc(sizeof(int   ) * (N+nClusters));
    cluster_end = (int*   ) malloc(sizeof(int   ) * 2 * (nClusters+1));
    cluster_neighbors     = (int*) malloc(sizeof(int) * nNeighbors);
    cluster_neighbors_end = (int*) malloc(sizeof(int) * (nClusters+1));

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        x[i] = 0.0f;
        b[i] = bVec[vertexReordering[i]];
        r[i] = b[i];
        m[i] = (t < 0)?1e-12:mesh.vertexDualVolumes[vertexReordering[i]];
    }
    if (t < 0) t = 1;
    return 0;

    //double* cotans    = (double*)  localArray;
    //int*    neighbors = (int*)    &localArray[8 * edges_in_cluster];
    //int*    end       = (int*)    &localArray[8 * edges_in_cluster + 4 * edges_in_cluster];
    //double* p         = (double*) &localArray[8 * edges_in_cluster + 4 * edges_in_cluster + 4 * verts_in_cluster];
    // Data per block: edges_in_cluster doubles + edges_in_cluster ints + verts_in_cluster ints + verts_accessed_by_cluster doubles
    // = 12 * edges_in_cluster + 4 * verts_in_cluster + 8 * verts_accessed_by_cluster


    // Cluster end = [vertex end, edge end, vertex end, edge end, ...
    // Vertex end is in local indices
    size_t MAX_CLUSTER_DATA_SIZE = 0;
    int pos = 0;
    int vIdx = 0;
    int neighborPos = 0;
    cluster_end[0] = 0;
    cluster_end[1] = 0;
    cluster_neighbors_end[0] = 0;
    for (size_t iC = 0; iC < nClusters; ++iC) {
        std::vector<size_t> cluster = clusters[iC];
        int localPos = 0;
        int edges_in_cluster = 0;
        vertex_end[pos] = 0;
        for (size_t iV : cluster) {
            for (std::pair<size_t, double> elem : weights[iV]) {
                neighbors[pos] = localInverseReordering[iC][elem.first];
                cotans[pos] = elem.second;
                pos += 1;
                localPos += 1;
                edges_in_cluster += 1;
            }
            vertex_end[vIdx + iC + 1] = localPos;
            vIdx += 1;
        }
        cluster_end[2 * iC + 2] = vIdx;
        cluster_end[2 * iC + 3] = pos;

        for (size_t iN : clusterNeighbors[iC]) {
            cluster_neighbors[neighborPos] = iN;
            neighborPos++;
        }
        cluster_neighbors_end[iC+1] = neighborPos;
        MAX_CLUSTER_DATA_SIZE = std::max(MAX_CLUSTER_DATA_SIZE,
                12 * edges_in_cluster + 12 * cluster.size() + 8 * clusterNeighbors[iC].size());
    }
    printf("MAX_CLUSTER_DATA_SIZE: %d\n", (int) MAX_CLUSTER_DATA_SIZE);
    return 0;

    // Allocate device memory
    cudaMalloc((void**)&d_x,           sizeof(double) * N);
    cudaMalloc((void**)&d_b,           sizeof(double) * N);
    cudaMalloc((void**)&d_Ap,          sizeof(double) * N);
    cudaMalloc((void**)&d_p,           sizeof(double) * N);
    cudaMalloc((void**)&d_r,           sizeof(double) * N);
    cudaMalloc((void**)&d_m,           sizeof(double) * N);
    cudaMalloc((void**)&d_neighbors,   sizeof(int   ) * nEdges);
    cudaMalloc((void**)&d_cotans,      sizeof(double) * nEdges);
    cudaMalloc((void**)&d_vertex_end,  sizeof(int   ) * (N+nClusters));
    cudaMalloc((void**)&d_cluster_end, sizeof(int   ) * 2 * (nClusters+1));
    cudaMalloc((void**)&d_alpha,       sizeof(double) * 1);
    cudaMalloc((void**)&d_beta,        sizeof(double) * 1);
    cudaMalloc((void**)&d_old_r2,      sizeof(double) * 1);
    cudaMalloc((void**)&d_new_r2,      sizeof(double) * 1);
    cudaMalloc((void**)&d_pAp,         sizeof(double) * 1);
    cudaMalloc((void**)&d_cluster_neighbors,     sizeof(int   ) * nNeighbors);
    cudaMalloc((void**)&d_cluster_neighbors_end, sizeof(int   ) * (nClusters+1));

    // Transfer data from host to device memory
    cudaMemcpy(d_x,           x,           sizeof(double) * N,                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,           r,           sizeof(double) * N,                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,           b,           sizeof(double) * N,                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,           m,           sizeof(double) * N,                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors,   neighbors,   sizeof(int   ) * nEdges,            cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertex_end,  vertex_end,  sizeof(int   ) * (N+nClusters),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_end, cluster_end, sizeof(int   ) * 2 * (nClusters+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cotans,      cotans,      sizeof(double) * nEdges,            cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_neighbors,     cluster_neighbors,     sizeof(int) * nNeighbors,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_neighbors_end, cluster_neighbors_end, sizeof(int) * (nClusters+1),  cudaMemcpyHostToDevice);

    bool done = false;
    int iter = 0;


    computeApClusteredCSR<<<NBLOCK,NTHREAD, MAX_CLUSTER_DATA_SIZE>>>(d_Ap, d_b, d_cotans, d_neighbors, d_vertex_end,
                                     d_cluster_neighbors, d_cluster_neighbors_end, d_cluster_end, d_m, t, N);
    // Transfer data back to host memory
    cudaMemcpy(b, d_Ap, sizeof(double) * N, cudaMemcpyDeviceToHost);

    printf("here\n");
    for (int i = 0; i < N; ++i) {
      bVec[vertexReordering[i]] = b[i];
    }
    printf("there\n");

    // https://stackoverflow.com/questions/12400477/retaining-dot-product-on-gpgpu-using-cublas-routine/12401838#12401838
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    //__global__ void computeApClusteredCSR(double *out,
                                      //double *global_p,
                                      //double *global_cotans,
                                      //int* global_neighbors,
                                      //int* global_vertex_end,
                                      //int* global_cluster_neighbors,
                                      //int* global_cluster_neighbors_end,
                                      //int* cluster_end,
                                      //double* m,
                                      //double t,
                                      //int n) {

    computeApClusteredCSR<<<NBLOCK,NTHREAD, MAX_CLUSTER_DATA_SIZE>>>(d_Ap, d_x, d_cotans, d_neighbors, d_vertex_end,
                                     d_cluster_neighbors, d_cluster_neighbors_end, d_cluster_end, d_m, t, N);
    vector_sub<<<NBLOCK,NTHREAD>>>(d_r, d_b, d_Ap, N);
    vector_cpy<<<NBLOCK,NTHREAD>>>(d_p, d_r, N);
    cublasDdot(handle, N, d_r, 1, d_r, 1, d_old_r2);

    int substeps = 40;
    double norm = 0;
    while (!done) {
      for (int i = 0; i < substeps; ++i) {
        computeApClusteredCSR<<<NBLOCK,NTHREAD, MAX_CLUSTER_DATA_SIZE>>>(d_Ap, d_p, d_cotans, d_neighbors, d_vertex_end,
                                         d_cluster_neighbors, d_cluster_neighbors_end, d_cluster_end, d_m, t, N);
        cublasDdot(handle, N, d_p, 1, d_Ap, 1, d_pAp);
        div<<<NBLOCK, NTHREAD>>>(d_alpha, d_old_r2, d_pAp);
        update_x_r<<<NBLOCK, NTHREAD>>>(d_x, d_r, d_alpha, d_p, d_Ap, N);
        cublasDdot(handle, N, d_r, 1, d_r, 1, d_new_r2);
        div<<<NBLOCK, NTHREAD>>>(d_beta, d_new_r2, d_old_r2);
        update_p<<<NBLOCK, NTHREAD>>>(d_p, d_beta, d_r, N);
        vector_cpy<<<NBLOCK,NTHREAD>>>(d_old_r2, d_new_r2, 1);
      }

      // Transfer data back to host memory
      cudaMemcpy(r, d_r, sizeof(double) * N, cudaMemcpyDeviceToHost);
      norm = 0;
      for (int i = 0; i < N; i++) {
        norm = fmax(norm, fabs(r[i]));
      }
      ++iter;
      if (verbose) printf("%d: residual: %f\n", iter, norm);
      done = (norm < tol) || (iter > 300);
    }
    if (norm >= tol)
        printf("timed out :'(");
    printf("DOne solving\n");
    cublasDestroy(handle);

    // Transfer data back to host memory
    cudaMemcpy(x, d_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

    printf("here\n");
    for (int i = 0; i < N; ++i) {
      xOut[vertexReordering[i]] = x[i];
    }
    printf("there\n");

    // Deallocate device memory
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_Ap);
    cudaFree(d_p);
    cudaFree(d_r);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_old_r2);
    cudaFree(d_new_r2);
    cudaFree(d_neighbors);
    cudaFree(d_cotans);

    // Deallocate host memory
    free(x);
    free(b);
    free(cotans);
    free(neighbors);

    printf("returning\n");
    return 0;
}
