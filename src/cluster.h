#include "tet.h"
#include <vector>

namespace CompArch {
bool contains(std::vector<size_t> v, size_t x);

std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>>
axisAlignedCluster(const TetMesh& t, std::vector<size_t> activeVertices,
                   size_t clusterSize);

std::vector<std::vector<size_t>> cluster(const TetMesh& t, size_t clusterSize);

std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>>
    clusterAndNeighbors(const TetMesh& t, size_t clusterSize);
} // namespace CompArch
