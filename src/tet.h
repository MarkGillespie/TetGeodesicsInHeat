#include "geometrycentral/utilities/vector3.h"
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/SparseCore>

using geometrycentral::Vector3;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::istringstream;
using std::string;

namespace CompArch {
class Vertex {
  public:
    Vector3 position;
    std::vector<size_t> edges; // Indices of incident PartialEdges
};

class PartialEdge {
  public:
    size_t src, dst; // Indices of src and dst vertices
    size_t tet;      // Index of tet
};

class Tet {
  public:
    std::vector<size_t> verts; // indices of vertices
    std::vector<size_t> edges; // indices of PartialEdges
    std::vector<size_t> neigh; // indices of neighboring Tets
                               // neigh[i] is opposite verts[i]

    // Partial edges stored in order:
    //   for (size_t i = 0; i < 4; ++i) {
    //     for (size_t j = i+1; j < 4; ++j) {
    //        edge i <-> j
    //     }
    //   }
};

class TetMesh {
  public:
    std::vector<Vertex> vertices;
    std::vector<PartialEdge> edges;
    std::vector<Tet> tets;

    std::vector<double> scaleFactors;

    double tetVolume(Tet t);
    std::vector<double> dihedralAngles(Tet t);
    std::vector<double> cotanWeights(Tet t);
    /* std::vector<double> dihedralAngles(Tet t); */

    TetMesh();

    std::vector<glm::vec3> vertexPositions();
    std::vector<std::vector<size_t>> faceList();
    std::vector<std::vector<size_t>> tetList();

    static TetMesh* construct(const std::vector<Vector3>& positions,
                              const std::vector<std::vector<size_t>>& tets,
                              const std::vector<std::vector<size_t>>& neigh);

    void recomputeGeometry();

    std::vector<double> tetVolumes;
    std::vector<double> vertexDualVolumes;
    std::vector<double> partialEdgeCotanWeights;

    Eigen::SparseMatrix<double> weakLaplacian();
    Eigen::SparseMatrix<double> massMatrix();

    // Assumes that *.node file is at same place as *.ele file,
    // but just ends in node
    static TetMesh* loadFromFile(string elePath);
    static double intrinsicVolume(double U, double u, double V, double v,
                                  double W, double w);
};
} // namespace CompArch
