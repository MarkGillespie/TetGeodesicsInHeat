#pragma once

#include "vector3.h"
#include <cmath>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

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
    std::array<size_t, 4> verts; // indices of vertices
    std::array<size_t, 6> edges; // indices of PartialEdges
    std::array<int, 4> neigh;    // indices of neighboring Tets
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
    std::array<double, 6> dihedralAngles(Tet t);
    std::array<double, 6> cotanWeights(Tet t);
    double meanEdgeLength();
    /* std::vector<double> dihedralAngles(Tet t); */

    TetMesh();

    std::vector<glm::vec3> vertexPositions();
    std::vector<std::array<size_t, 3>> faceList();
    std::vector<std::array<size_t, 4>> tetList();
    std::vector<int> neighborList();

    static TetMesh* construct(const std::vector<Vector3>& positions,
                              const std::vector<std::array<size_t, 4>>& tets,
                              const std::vector<std::array<int, 4>>& neigh);

    void recomputeGeometry();

    std::vector<double> tetVolumes;
    std::vector<double> vertexDualVolumes;
    std::vector<double> faceAreas;
    std::vector<double> partialEdgeCotanWeights;

    Eigen::SparseMatrix<double> weakLaplacian();
    Eigen::SparseMatrix<double> massMatrix();

    std::vector<double> distances(std::vector<double> start, double t,
                                  bool verbose = false);

    std::array<Vector3, 4> layOutIntrinsicTet(Tet t);

    // Assumes that *.node file is at same place as *.ele file,
    // but just ends in node
    static TetMesh* loadFromFile(string elePath);
    static double intrinsicVolume(double U, double u, double V, double v,
                                  double W, double w);
};

// return the gradient of function u linearly interpolated over a tetrahedron
// with vertices p[0], ... , p[3]
Vector3 grad(std::array<double, 4> u, std::array<Vector3, 4> p);

// returns the integrated divergence of vector field X associated with
// each vertex p[i] of a tetrahedron
std::array<double, 4> div(Vector3 X, std::array<Vector3, 4> p);

} // namespace CompArch
