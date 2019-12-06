#include "cluster.h"
namespace CompArch {

Vector3 extent(const TetMesh& t, std::vector<size_t> activeVertices) {
    Vector3 min = t.vertices[activeVertices[0]].position;
    Vector3 max = t.vertices[activeVertices[0]].position;

    for (size_t iV : activeVertices) {
        Vector3 p = t.vertices[iV].position;
        min       = componentwiseMin(min, p);
        max       = componentwiseMax(max, p);
    }

    return max - min;
}

bool contains(std::vector<size_t> v, size_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

size_t neighborhoodSize(const TetMesh& t, std::vector<size_t> activeVertices, std::vector<size_t>& neighbors) {
    // Neighborhood contains all active vertices and all neighbors of active
    // vertices
    size_t neighborhoodSize = activeVertices.size();
    for (size_t iV : activeVertices) {
        for (size_t iE : t.vertices[iV].edges) {
            PartialEdge e = t.edges[iE];
            if (!contains(activeVertices, e.src) &&
                !contains(neighbors, e.src)) {
                neighborhoodSize++;
                neighbors.push_back(e.src);
            } else if (!contains(activeVertices, e.dst) &&
                       !contains(neighbors, e.dst)) {
                neighborhoodSize++;
                neighbors.push_back(e.dst);
            }
        }
    }
    return neighborhoodSize;
}

std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>>
axisAlignedCluster(const TetMesh& t, std::vector<size_t> activeVertices,
                   size_t clusterSize) {

    std::vector<size_t> neighbors;
    if (neighborhoodSize(t, activeVertices, neighbors) < clusterSize) {
        std::vector<std::vector<size_t>> singleCluster, singleNeighbors;
        singleCluster.push_back(activeVertices);
        singleNeighbors.push_back(neighbors);
        return std::make_pair(singleCluster, singleNeighbors);
    }

    Vector3 diag = extent(t, activeVertices);

    size_t splitAxis;

    if (diag.x >= diag.y && diag.x >= diag.z) { /* Split along x axis */
        splitAxis = 0;
    } else if (diag.y >= diag.x && diag.y >= diag.z) { /* Split along y axis */
        splitAxis = 1;
    } else if (diag.z >= diag.x && diag.z >= diag.y) { /* Split along z axis */
        splitAxis = 2;
    }

    std::vector<std::pair<Vector3, size_t>> taggedPositions;
    taggedPositions.reserve(activeVertices.size());
    for (size_t iV : activeVertices) {
        taggedPositions.push_back(std::make_pair(t.vertices[iV].position, iV));
    }

    auto axisCompare = [&](std::pair<Vector3, size_t> x,
                           std::pair<Vector3, size_t> y) {
        return x.first[splitAxis] < y.first[splitAxis];
    };

    // https://stackoverflow.com/questions/42791860/finding-the-median-value-of-a-vector-using-c/42791986
    std::nth_element(taggedPositions.begin(),
                     taggedPositions.begin() + taggedPositions.size() / 2,
                     taggedPositions.end(), axisCompare);

    std::vector<std::pair<Vector3, size_t>> taggedLower(
        taggedPositions.begin(),
        taggedPositions.begin() + taggedPositions.size() / 2);
    std::vector<std::pair<Vector3, size_t>> taggedUpper(
        taggedPositions.begin() + taggedPositions.size() / 2,
        taggedPositions.end());
    std::vector<size_t> lower, upper;
    for (auto p : taggedLower) lower.push_back(p.second);
    for (auto p : taggedUpper) upper.push_back(p.second);

    auto lowerClusters = axisAlignedCluster(t, lower, clusterSize);
    auto upperClusters = axisAlignedCluster(t, upper, clusterSize);

    lowerClusters.first.insert(lowerClusters.first.end(), upperClusters.first.begin(),
                         upperClusters.first.end());
    lowerClusters.second.insert(lowerClusters.second.end(), upperClusters.second.begin(),
                         upperClusters.second.end());

    return lowerClusters;
}

std::vector<std::vector<size_t>> cluster(const TetMesh& t, size_t clusterSize) {
    return clusterAndNeighbors(t, clusterSize).first;
}

std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>>
    clusterAndNeighbors(const TetMesh& t, size_t clusterSize) {

    std::vector<size_t> activeVertices;
    activeVertices.reserve(t.vertices.size());
    for (size_t i = 0; i < t.vertices.size(); ++i) activeVertices.push_back(i);
    return axisAlignedCluster(t, activeVertices, clusterSize);
}

} // namespace CompArch
