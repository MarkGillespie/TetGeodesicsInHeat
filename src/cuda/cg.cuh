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

using namespace CompArch;

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
int cgSolve(Eigen::VectorXd& xOut, Eigen::VectorXd b, const TetMesh& mesh, double tol=1e-8, double t = -1, bool verbose = false);
