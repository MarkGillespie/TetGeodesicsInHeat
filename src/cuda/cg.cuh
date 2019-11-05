#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../tet.h"

namespace CompArch {

std::vector<double> solveCG(const TetMesh& mesh, std::vector<double> rhs);

} // CompArch