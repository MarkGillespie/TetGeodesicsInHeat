#include "tet.h"
#include <vector>

namespace CompArch {
std::vector<std::vector<size_t>>
axisAlignedCluster(TetMesh& t, std::vector<size_t> activeVertices,
                   size_t clusterSize);
std::vector<std::vector<size_t>> cluster(TetMesh& t, size_t clusterSize);
} // namespace CompArch
