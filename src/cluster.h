#include "tet.h"
#include <vector>

namespace CompArch{
std::vector<size_t> cluster(TetMesh t, size_t clusterSize);
std::vector<size_t> axisAlignedCluster(std::vector<Vector3> positions, size_t clusterSize, size_t& nClusters);
} // CompArch
