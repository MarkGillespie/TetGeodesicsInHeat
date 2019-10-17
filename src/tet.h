#include "geometrycentral/utilities/vector3.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using std::string;
using std::ifstream;
using std::istringstream;
using std::cout;
using std::cerr;
using std::endl;
using geometrycentral::Vector3;

class Vertex {
    public:
        Vector3 position;
        std::vector<size_t> edges; // Indices of incident PartialEdges
};

class PartialEdge {
    public:
        size_t src, dst; // Indices of src and dst vertices
        size_t tet; // Index of tet
};

class Tet {
    public:
        Tet();

        std::vector<size_t> verts; // Vertices as a set
        const size_t& a, b, c, d; // Indices of Vertices
        size_t ab, ac, ad, bc, bd, cd; // Indices of PartialEdges
        std::vector<size_t> neigh;
        const size_t& A, B, C, D; // Indices of neighboring tets
};

class TetMesh {
    public:
        std::vector<Vertex> vertices;
        std::vector<PartialEdge> edges;
        std::vector<Tet> tets;

        TetMesh();

        static TetMesh* construct(const std::vector<Vector3>& positions,
                const std::vector<std::vector<size_t>>& tets, const std::vector<std::vector<size_t>>& neigh);

        // Assumes that *.node file is at same place as *.ele file,
        // but just ends in node
        static TetMesh* loadFromFile(string elePath);
};
