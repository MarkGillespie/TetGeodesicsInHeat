#include "cluster.h"
namespace CompArch {

Vector3 extent(std::vector<Vector3> ps) {
    Vector3 min = ps[0];
    Vector3 max = ps[0];

    for (auto p : ps) {
        min = componentwiseMin(min, p);
        max = componentwiseMax(max, p);
    }

    return max - min;
}

std::vector<size_t> axisAlignedCluster(std::vector<Vector3> positions,
                                       size_t clusterSize, size_t& nClusters) {

    if (positions.size() < clusterSize) {
        nClusters = 1;
        return std::vector<size_t>(positions.size(), 0); // All elements in cluster 0
    }

    Vector3 diag = extent(positions);

    size_t splitAxis;
    if (diag.x >= diag.y && diag.x >= diag.z) {
        splitAxis = 0; // Split along x axis
    } else if (diag.y >= diag.x && diag.y >= diag.z) { splitAxis = 1; // Split along y axis
    } else if (diag.z >= diag.x && diag.z >= diag.y) {
        splitAxis = 2; // Split along z axis
    }

    std::vector<std::pair<Vector3, size_t>> taggedPositions;
    taggedPositions.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        taggedPositions.push_back(std::make_pair(positions[i], i));
    }

    auto axisCompare = [&](std::pair<Vector3, size_t> x, std::pair<Vector3, size_t> y) {
        return x.first[splitAxis] < y.first[splitAxis];
    };

    // https://stackoverflow.com/questions/42791860/finding-the-median-value-of-a-vector-using-c/42791986
    std::nth_element(taggedPositions.begin(),
                     taggedPositions.begin() + taggedPositions.size()/2,
                     taggedPositions.end(),
                     axisCompare);

    std::vector<std::pair<Vector3, size_t>> taggedLower(taggedPositions.begin(),
                                                        taggedPositions.begin() + taggedPositions.size()/2);
    std::vector<std::pair<Vector3, size_t>> taggedUpper(taggedPositions.begin() + taggedPositions.size()/2,
                                                        taggedPositions.end());
    std::vector<Vector3> lower, upper;
    for (auto p : taggedLower) lower.push_back(p.first);
    for (auto p : taggedUpper) upper.push_back(p.first);

    size_t nLowerClusters, nUpperClusters;
    std::vector<size_t> lowerClusters = axisAlignedCluster(lower, clusterSize, nLowerClusters);
    std::vector<size_t> upperClusters = axisAlignedCluster(upper, clusterSize, nUpperClusters);

    nClusters = nLowerClusters + nUpperClusters;
    std::vector<size_t> clusters(positions.size(), 0);
    for (size_t i = 0; i < lowerClusters.size(); ++i) {
        size_t vIdx = taggedLower[i].second;
        clusters[vIdx] = lowerClusters[i];
    }
    for (size_t i = 0; i < upperClusters.size(); ++i) {
        size_t vIdx = taggedUpper[i].second;
        clusters[vIdx] = nLowerClusters + upperClusters[i];
    }

    return clusters;
}

std::vector<size_t> cluster(TetMesh t, size_t clusterSize) {
    size_t nClusters = 0;
    return axisAlignedCluster(t.vertexPositions(), clusterSize, nClusters);
}

} // CompArch
