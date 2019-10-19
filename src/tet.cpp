#include "tet.h"
namespace CompArch {
TetMesh::TetMesh(){
  recomputeGeometry();
}

// https://people.eecs.berkeley.edu/~wkahan/VtetLang.pdf
// https://www.geeksforgeeks.org/program-to-find-the-volume-of-an-irregular-tetrahedron/
double TetMesh::intrinsicVolume(double U, double u, double V, double v, double W, double w) {
    // Steps to calculate volume of a
    // Tetrahedron using formula
    double uPow = pow(u, 2);
    double vPow = pow(v, 2);
    double wPow = pow(w, 2);
    double UPow = pow(U, 2);
    double VPow = pow(V, 2);
    double WPow = pow(W, 2);

    double a = 4 * uPow * vPow * wPow
      - uPow * pow((vPow + wPow - UPow), 2)
      - vPow * pow((wPow + uPow - VPow), 2)
      - wPow * pow((uPow + vPow - WPow), 2)
      + (vPow + wPow - UPow) * (wPow + uPow - VPow)
      * (uPow + vPow - WPow);
    return sqrt(a) / 12;
}

double TetMesh::tetVolume(Tet t) {
  Vector3 p0 = vertices[t.verts[0]].position;
  Vector3 p1 = vertices[t.verts[1]].position;
  Vector3 p2 = vertices[t.verts[2]].position;
  Vector3 p3 = vertices[t.verts[3]].position;

  double u0 = scaleFactors[t.verts[0]];
  double u1 = scaleFactors[t.verts[1]];
  double u2 = scaleFactors[t.verts[2]];
  double u3 = scaleFactors[t.verts[3]];

  double U = norm(p0 - p1) * exp(0.5 * (u0 + u1));
  double u = norm(p2 - p3) * exp(0.5 * (u2 + u3));
  double V = norm(p0 - p2) * exp(0.5 * (u0 + u2));
  double v = norm(p1 - p3) * exp(0.5 * (u1 + u3));
  double W = norm(p1 - p2) * exp(0.5 * (u1 + u2));
  double w = norm(p0 - p3) * exp(0.5 * (u0 + u3));

  return intrinsicVolume(U, u, V, v, W, w);
}

void TetMesh::recomputeGeometry() {

  // Loop over tets to compute cotan weights and vertex areas
  vertexDualVolumes = std::vector<double>(vertices.size(), 0.0);
  for (size_t iT = 0; iT < tets.size(); ++iT) {

    double vol = tetVolume(tets[iT]);
    tetVolumes[iT] = vol;
    for (size_t i = 0; i < 4; ++i) {
      vertexDualVolumes[tets[iT].verts[i]] += vol / 4;
    }
  }
}

std::vector<glm::vec3> TetMesh::vertexPositions() {
    std::vector<glm::vec3> vertexPositions;
    for (Vertex v : vertices) {
        vertexPositions.emplace_back(glm::vec3{v.position.x, v.position.y, v.position.z});
    }

    return vertexPositions;
}

std::vector<std::vector<size_t>> TetMesh::faceList() {
    std::vector<std::vector<size_t>> faces;
    for (Tet t : tets) {
        faces.emplace_back(std::vector<size_t>{t.verts[0], t.verts[1], t.verts[2]});
        faces.emplace_back(std::vector<size_t>{t.verts[0], t.verts[2], t.verts[3]});
        faces.emplace_back(std::vector<size_t>{t.verts[0], t.verts[3], t.verts[1]});
        faces.emplace_back(std::vector<size_t>{t.verts[2], t.verts[1], t.verts[3]});
    }

    return faces;
}

std::vector<std::vector<size_t>> TetMesh::tetList() {
    std::vector<std::vector<size_t>> tetCombinatorics;
    for (Tet t : tets) {
        tetCombinatorics.emplace_back(t.verts);
    }

    return tetCombinatorics;
}

TetMesh* TetMesh::construct(const std::vector<Vector3>& positions,
            const std::vector<std::vector<size_t>>& tets,
            const std::vector<std::vector<size_t>>& neigh) {
    TetMesh* mesh = new TetMesh();

    for (Vector3 p : positions) {
        Vertex v;
        v.position = p;
        mesh->vertices.emplace_back(v);
    }
    mesh->scaleFactors = std::vector<double>(mesh->vertices.size(), 0.0);

    for (size_t n = 0; n < tets.size(); ++n) {
        Tet t;
        t.verts = tets[n];
        // reorder vertices to be positively oriented
        // the tetrahedron is positively oriented if it has positive volume
        Vector3 v0 = mesh->vertices[t.verts[0]].position;
        Vector3 v1 = mesh->vertices[t.verts[1]].position;
        Vector3 v2 = mesh->vertices[t.verts[2]].position;
        Vector3 v3 = mesh->vertices[t.verts[3]].position;
        double vol = dot(v3, cross(v1 - v0, v2 - v0));
        if (vol < 0) {
            // Have to reverse orientation.
            // We do so by swapping v1 and v2
            size_t temp = t.verts[0];
            t.verts[0] = t.verts[1];
            t.verts[1] = temp;
        }

        // TODO: Order the neighbors so that neigh[i] is opposite verts[i]
        t.neigh = neigh[n];

        // Add in PartialEdges
        size_t tIdx = tets.size();
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = i+1; j < 4; ++j) {
                size_t eIdx = mesh->edges.size();
                t.edges.emplace_back(eIdx);
                mesh->vertices[t.verts[i]].edges.emplace_back(eIdx);
                mesh->vertices[t.verts[j]].edges.emplace_back(eIdx);
                mesh->edges.emplace_back(PartialEdge{t.verts[i], t.verts[j], tIdx});
            }
        }
        mesh->tets.emplace_back(t);
    }

    return mesh;
}

TetMesh* TetMesh::loadFromFile(string elePath) {
    string extension = elePath.substr(elePath.find_last_of(".") + 1);
    if (extension != "ele") {
        cerr << "Error: " << elePath << "is not a *.ele file. It has extension '" << extension  << "'" << endl;
    }

    string nodePath = elePath.substr(0, elePath.find_last_of(".")) + ".node";
    string neighPath = elePath.substr(0, elePath.find_last_of(".")) + ".neigh";


    std::vector<Vector3> verts;
    ifstream node(nodePath);
    if (node.is_open()) {
        string line;
        getline(node, line);
        while (getline(node, line)) {
            if (line[0] == '#' || line.length() == 0) continue;

            istringstream ss(line);
            size_t idx;
            double x, y, z;
            ss >> idx >> x >> y >> z;
            verts.emplace_back(Vector3{x, y, z});
            assert(idx == verts.size());
        }

        node.close();
    }

    std::vector<std::vector<size_t>> tets;
    ifstream ele(elePath);
    if (ele.is_open()) {
        string line;
        getline(ele, line);
        while (getline(ele, line)) {
            if (line[0] == '#' || line.length() == 0) continue;

            istringstream ss(line);
            size_t idx, a, b, c, d;
            ss >> idx >> a >> b >> c >> d;
            assert(a <= verts.size());
            assert(b <= verts.size());
            assert(c <= verts.size());
            assert(d <= verts.size());
            assert(a > 0);
            assert(b > 0);
            assert(c > 0);
            assert(d > 0);

            // 1-indexed?
            tets.emplace_back(std::vector<size_t>{a - 1, b - 1, c - 1, d - 1});
            assert(a-1 < verts.size());
            assert(b-1 < verts.size());
            assert(c-1 < verts.size());
            assert(d-1 < verts.size());
            //tets.emplace_back(std::vector<size_t>{a , b , c , d });
            assert(idx == tets.size());
        }

        ele.close();
    }

    std::vector<std::vector<size_t>> neighbors;
    ifstream neigh(neighPath);
    if (neigh.is_open()) {
        string line;
        getline(neigh, line);
        while (getline(neigh, line)) {
            if (line[0] == '#' || line.length() == 0) continue;

            istringstream ss(line);
            size_t idx, a, b, c, d;
            ss >> idx >> a >> b >> c >> d;
            neighbors.emplace_back(std::vector<size_t>{a, b, c, d});
            assert(idx == neighbors.size());
        }

        neigh.close();
    }

    return TetMesh::construct(verts, tets, neighbors);
}
} // CompArch
