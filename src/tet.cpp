#include "tet.h"

namespace CompArch {
TetMesh::TetMesh() {}

Eigen::SparseMatrix<double> TetMesh::massMatrix() {
    Eigen::SparseMatrix<double> M(vertices.size(), vertices.size());
    std::vector<Eigen::Triplet<double>> tripletList;

    for (size_t i = 0; i < vertexDualVolumes.size(); ++i) {
        tripletList.emplace_back(i, i, vertexDualVolumes[i]);
    }

    M.setFromTriplets(tripletList.begin(), tripletList.end());
    return M;
}

Eigen::SparseMatrix<double> TetMesh::weakLaplacian() {
    Eigen::SparseMatrix<double> cotanLaplacian(vertices.size(),
                                               vertices.size());
    std::vector<Eigen::Triplet<double>> tripletList;
    assert(partialEdgeCotanWeights.size() == edges.size());

    for (size_t iPE = 0; iPE < edges.size(); ++iPE) {
        PartialEdge pe = edges[iPE];
        size_t vSrc    = pe.src;
        size_t vDst    = pe.dst;

        double weight = partialEdgeCotanWeights[iPE];

        tripletList.emplace_back(vSrc, vSrc, weight);
        tripletList.emplace_back(vDst, vDst, weight);
        tripletList.emplace_back(vSrc, vDst, -weight);
        tripletList.emplace_back(vDst, vSrc, -weight);
    }

    cotanLaplacian.setFromTriplets(tripletList.begin(), tripletList.end());
    return cotanLaplacian;
}


// return the gradient of function u linearly interpolated over a tetrahedron
// with vertices p[0], ... , p[3]
Vector3 grad(std::vector<double> u, std::vector<Vector3> p) {
    Vector3 gradU;
    double vol = dot(p[3] - p[0], cross(p[1] - p[0], p[2] - p[0]));

    // Hand code cases because of orientation problems
    // Face 012
    Vector3 areaNormal = -0.5 * cross(p[1] - p[0], p[2] - p[0]);
    gradU += u[3] * areaNormal / (3 * vol);
    // Face 023
    areaNormal = -0.5 * cross(p[2] - p[0], p[3] - p[0]);
    gradU += u[1] * areaNormal / (3 * vol);
    // Face 031
    areaNormal = -0.5 * cross(p[3] - p[0], p[1] - p[0]);
    gradU += u[2] * areaNormal / (3 * vol);
    // Face 213
    areaNormal = -0.5 * cross(p[1] - p[2], p[3] - p[2]);
    gradU += u[0] * areaNormal / (3 * vol);

    return gradU;
}

// Returns the angle of the corner opposite edge a in a triangle
// with edge lengths a, b, c
double cornerAngle(double a, double b, double c) {
    double cosAngle = (b * b + c * c - a * a) / (2 * b * c);
    return acos(cosAngle);
}

// We will put vertex 0 at the origin and vertex 1 along the x axis
// We put vertex 2 somewhere in the x-y plane, and vertex 3 somewhere
// in space
// List returned in vertex order
std::vector<Vector3> TetMesh::layOutIntrinsicTet(Tet t) {
    std::vector<Vector3> positions;

    Vector3 p0 = vertices[t.verts[0]].position;
    Vector3 p1 = vertices[t.verts[1]].position;
    Vector3 p2 = vertices[t.verts[2]].position;
    Vector3 p3 = vertices[t.verts[3]].position;

    double u0 = scaleFactors[t.verts[0]];
    double u1 = scaleFactors[t.verts[1]];
    double u2 = scaleFactors[t.verts[2]];
    double u3 = scaleFactors[t.verts[3]];

    double e01 = norm(p0 - p1) * exp(0.5 * (u0 + u1));
    double e02 = norm(p0 - p2) * exp(0.5 * (u0 + u2));
    double e03 = norm(p0 - p3) * exp(0.5 * (u0 + u3));
    double e12 = norm(p1 - p2) * exp(0.5 * (u1 + u2));
    double e13 = norm(p1 - p3) * exp(0.5 * (u1 + u3));
    double e23 = norm(p2 - p3) * exp(0.5 * (u2 + u3));

    positions.emplace_back(Vector3{0, 0, 0});
    positions.emplace_back(Vector3{e01, 0, 0});
    double angle = cornerAngle(e12, e01, e02);
    positions.emplace_back(e02 * Vector3{cos(angle), -sin(angle), 0});

    // To place vertex 3, we find the angle at the corner of v0 in face v0-v1-v3
    // Together with the dihedral angle between the 013 face and the 012 face
    // (i.e. the dihedral angle along edge e01), this tells us where v3 should
    // go
    angle                = cornerAngle(e13, e01, e03);
    double dihedralAngle = dihedralAngles(t)[0];
    positions.emplace_back(e03 * Vector3{cos(angle),
                                         -cos(dihedralAngle) * sin(angle),
                                         sin(dihedralAngle) * sin(angle)});


    return positions;
}

// https://people.eecs.berkeley.edu/~wkahan/VtetLang.pdf
// https://www.geeksforgeeks.org/program-to-find-the-volume-of-an-irregular-tetrahedron/
double TetMesh::intrinsicVolume(double U, double u, double V, double v,
                                double W, double w) {
    // Steps to calculate volume of a
    // Tetrahedron using formula
    double uPow = pow(u, 2);
    double vPow = pow(v, 2);
    double wPow = pow(w, 2);
    double UPow = pow(U, 2);
    double VPow = pow(V, 2);
    double WPow = pow(W, 2);

    double a =
        4 * uPow * vPow * wPow - uPow * pow((vPow + wPow - UPow), 2) -
        vPow * pow((wPow + uPow - VPow), 2) -
        wPow * pow((uPow + vPow - WPow), 2) +
        (vPow + wPow - UPow) * (wPow + uPow - VPow) * (uPow + vPow - WPow);
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

double area(double a, double b, double c) {
    double s = 0.5 * (a + b + c);
    return sqrt(s * (s - a) * (s - b) * (s - c));
}

// Relations between edge lengths, dihedral and solid angles in tetrahedra
// Wirth and Dreiding
std::vector<double> TetMesh::dihedralAngles(Tet t) {
    Vector3 p0 = vertices[t.verts[0]].position;
    Vector3 p1 = vertices[t.verts[1]].position;
    Vector3 p2 = vertices[t.verts[2]].position;
    Vector3 p3 = vertices[t.verts[3]].position;

    double u0 = scaleFactors[t.verts[0]];
    double u1 = scaleFactors[t.verts[1]];
    double u2 = scaleFactors[t.verts[2]];
    double u3 = scaleFactors[t.verts[3]];

    double e01 = norm(p0 - p1) * exp(0.5 * (u0 + u1));
    double e02 = norm(p0 - p2) * exp(0.5 * (u0 + u2));
    double e03 = norm(p0 - p3) * exp(0.5 * (u0 + u3));
    double e12 = norm(p1 - p2) * exp(0.5 * (u1 + u2));
    double e13 = norm(p1 - p3) * exp(0.5 * (u1 + u3));
    double e23 = norm(p2 - p3) * exp(0.5 * (u2 + u3));

    double e01sq = pow(e01, 2);
    double e02sq = pow(e02, 2);
    double e03sq = pow(e03, 2);
    double e12sq = pow(e12, 2);
    double e13sq = pow(e13, 2);
    double e23sq = pow(e23, 2);

    double d01 = -e01sq * e01sq +
                 e01sq * (e02sq + e03sq + e12sq + e13sq - 2 * e23sq) +
                 (e02sq - e12sq) * (e13sq - e03sq);
    double d02 = -e02sq * e02sq +
                 e02sq * (e01sq + e03sq + e12sq + e23sq - 2 * e13sq) +
                 (e01sq - e12sq) * (e23sq - e03sq);
    double d03 = -e03sq * e03sq +
                 e03sq * (e01sq + e02sq + e13sq + e23sq - 2 * e12sq) +
                 (e01sq - e13sq) * (e23sq - e02sq);
    double d12 = -e12sq * e12sq +
                 e12sq * (e01sq + e13sq + e02sq + e23sq - 2 * e03sq) +
                 (e01sq - e02sq) * (e23sq - e13sq);
    double d13 = -e13sq * e13sq +
                 e13sq * (e01sq + e12sq + e03sq + e23sq - 2 * e02sq) +
                 (e01sq - e03sq) * (e23sq - e12sq);
    double d23 = -e23sq * e23sq +
                 e23sq * (e02sq + e12sq + e03sq + e13sq - 2 * e01sq) +
                 (e02sq - e03sq) * (e13sq - e12sq);

    double d012 = -(e01 + e02 + e12) * (e01 + e02 - e12) * (e12 + e01 - e02) *
                  (e02 + e12 - e01);
    double d013 = -(e01 + e03 + e13) * (e01 + e03 - e13) * (e13 + e01 - e03) *
                  (e03 + e13 - e01);
    double d023 = -(e02 + e03 + e23) * (e02 + e03 - e23) * (e23 + e02 - e03) *
                  (e03 + e23 - e02);
    double d123 = -(e13 + e23 + e12) * (e13 + e23 - e12) * (e12 + e13 - e23) *
                  (e23 + e12 - e13);

    double cos01 = d01 / sqrt(d012 * d013);
    double cos02 = d02 / sqrt(d012 * d023);
    double cos03 = d03 / sqrt(d013 * d023);
    double cos12 = d12 / sqrt(d012 * d123);
    double cos13 = d13 / sqrt(d013 * d123);
    double cos23 = d23 / sqrt(d023 * d123);

    std::vector<double> angles;
    angles.emplace_back(acos(cos01));
    angles.emplace_back(acos(cos02));
    angles.emplace_back(acos(cos03));
    angles.emplace_back(acos(cos12));
    angles.emplace_back(acos(cos13));
    angles.emplace_back(acos(cos23));

    return angles;
}

double cot(double theta) { return 1 / tan(theta); }

std::vector<double> TetMesh::cotanWeights(Tet t) {
    std::vector<double> angles = dihedralAngles(t);

    Vector3 p0 = vertices[t.verts[0]].position;
    Vector3 p1 = vertices[t.verts[1]].position;
    Vector3 p2 = vertices[t.verts[2]].position;
    Vector3 p3 = vertices[t.verts[3]].position;

    double u0 = scaleFactors[t.verts[0]];
    double u1 = scaleFactors[t.verts[1]];
    double u2 = scaleFactors[t.verts[2]];
    double u3 = scaleFactors[t.verts[3]];

    double e01 = norm(p0 - p1) * exp(0.5 * (u0 + u1));
    double e02 = norm(p0 - p2) * exp(0.5 * (u0 + u2));
    double e03 = norm(p0 - p3) * exp(0.5 * (u0 + u3));
    double e12 = norm(p1 - p2) * exp(0.5 * (u1 + u2));
    double e13 = norm(p1 - p3) * exp(0.5 * (u1 + u3));
    double e23 = norm(p2 - p3) * exp(0.5 * (u2 + u3));

    std::vector<double> weights;
    // Weird order since the cotan weight for edge e_ij comes from the
    // _opposite_ edge e_kl
    weights.emplace_back(e23 * cot(angles[5]) / 6);
    weights.emplace_back(e13 * cot(angles[4]) / 6);
    weights.emplace_back(e12 * cot(angles[3]) / 6);
    weights.emplace_back(e03 * cot(angles[2]) / 6);
    weights.emplace_back(e02 * cot(angles[1]) / 6);
    weights.emplace_back(e01 * cot(angles[0]) / 6);
    return weights;
}


// Relations between edge lengths, dihedral and solid angles in tetrahedra
// Wirth and Dreiding
void TetMesh::recomputeGeometry() {

    // Loop over tets to compute cotan weights and vertex areas
    vertexDualVolumes = std::vector<double>(vertices.size(), 0.0);
    tetVolumes        = std::vector<double>(tets.size(), 0.0);
    for (size_t iT = 0; iT < tets.size(); ++iT) {
        double vol     = tetVolume(tets[iT]);
        tetVolumes[iT] = vol;
        for (size_t i = 0; i < 4; ++i) {
            vertexDualVolumes[tets[iT].verts[i]] += vol / 4;
        }

        std::vector<double> tetWeights = cotanWeights(tets[iT]);
        partialEdgeCotanWeights.insert(partialEdgeCotanWeights.end(),
                                       tetWeights.begin(), tetWeights.end());
    }
}

std::vector<glm::vec3> TetMesh::vertexPositions() {
    std::vector<glm::vec3> vertexPositions;
    // for (Vertex v : vertices) {
    for (size_t i = 0; i < vertices.size(); ++i) {
        Vertex v = vertices[i];
        vertexPositions.emplace_back(
            glm::vec3{v.position.x, v.position.y, v.position.z});
    }

    return vertexPositions;
}

std::vector<std::vector<size_t>> TetMesh::faceList() {
    std::vector<std::vector<size_t>> faces;
    for (Tet t : tets) {
        faces.emplace_back(
            std::vector<size_t>{t.verts[0], t.verts[1], t.verts[2]});
        faces.emplace_back(
            std::vector<size_t>{t.verts[0], t.verts[2], t.verts[3]});
        faces.emplace_back(
            std::vector<size_t>{t.verts[0], t.verts[3], t.verts[1]});
        faces.emplace_back(
            std::vector<size_t>{t.verts[2], t.verts[1], t.verts[3]});
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
        double vol = dot(v3 - v0, cross(v1 - v0, v2 - v0));
        if (vol > 0) {
            // Have to reverse orientation.
            // We do so by swapping v1 and v2
            size_t temp = t.verts[0];
            t.verts[0]  = t.verts[1];
            t.verts[1]  = temp;
        }

        // TODO: Order the neighbors so that neigh[i] is opposite verts[i]
        t.neigh = neigh[n];

        // Add in PartialEdges
        size_t tIdx = tets.size();
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = i + 1; j < 4; ++j) {
                size_t eIdx = mesh->edges.size();
                t.edges.emplace_back(eIdx);
                mesh->vertices[t.verts[i]].edges.emplace_back(eIdx);
                mesh->vertices[t.verts[j]].edges.emplace_back(eIdx);
                mesh->edges.emplace_back(
                    PartialEdge{t.verts[i], t.verts[j], tIdx});
            }
        }
        mesh->tets.emplace_back(t);
    }

    mesh->recomputeGeometry();
    return mesh;
}

TetMesh* TetMesh::loadFromFile(string elePath) {
    string extension = elePath.substr(elePath.find_last_of(".") + 1);
    if (extension != "ele") {
        cerr << "Error: " << elePath
             << "is not a *.ele file. It has extension '" << extension << "'"
             << endl;
    }

    string nodePath  = elePath.substr(0, elePath.find_last_of(".")) + ".node";
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
            assert(a - 1 < verts.size());
            assert(b - 1 < verts.size());
            assert(c - 1 < verts.size());
            assert(d - 1 < verts.size());
            // tets.emplace_back(std::vector<size_t>{a , b , c , d });
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
} // namespace CompArch
