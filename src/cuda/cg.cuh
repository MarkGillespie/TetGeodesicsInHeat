#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

#include "../tet.h"

using namespace CompArch;

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
void  cgSolve(Eigen::VectorXd& xOut, Eigen::VectorXd b, const TetMesh& mesh, double t = -1);
