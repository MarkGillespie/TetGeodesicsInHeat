#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>

#include "../tet.h"

using namespace CompArch;

#define NTHREAD 256
#define NBLOCK  5000

// If t > 0, solves (M + tL) x = b
// If t < 0, solves Lx = b
void  cgSolve(Eigen::VectorXd& xOut, Eigen::VectorXd b, const TetMesh& mesh, double t = -1);
