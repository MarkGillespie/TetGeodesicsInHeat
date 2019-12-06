#pragma once 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>


#include "../tet.h"
#include "../cluster.h"

using namespace CompArch;

std::vector<std::unordered_map<size_t, double>> edgeWeights(const TetMesh& mesh);

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
int cgSolve(Eigen::VectorXd& xOut, const Eigen::VectorXd b, const TetMesh& mesh, double tol=1e-8, double t = -1, bool verbose = false);

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
// Stores matrix in CSR format
int cgSolveCSR(Eigen::VectorXd& xOut, const Eigen::VectorXd b, const TetMesh& mesh, double tol, double t, bool verbose, std::vector<size_t> vertexPermutation);

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
// Stores matrix in CSR format
int cgSolveCSR(Eigen::VectorXd& xOut, const Eigen::VectorXd b, const TetMesh& mesh, double tol=1e-8, double t = -1, bool verbose = false, bool degreeSort = false);

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
// Stores matrix in CSR format
int cgSolveClusteredCSR(Eigen::VectorXd& xOut, const Eigen::VectorXd b, const TetMesh& mesh, double tol, double t, bool verbose);

