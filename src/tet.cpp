#include "tet.h"

namespace CompArch {

std::vector<double> TetMesh::distances(std::vector<double> start, double t, bool verbose) {
    if (verbose) cout << "Starting distance computation";
    Eigen::VectorXd u0 = Eigen::VectorXd::Map(start.data(), start.size());
    if (verbose) cout << "Building L";
    Eigen::SparseMatrix<double> L    = weakLaplacian();
    if (verbose) cout << "Computed L" << endl;
    Eigen::SparseMatrix<double> M    = massMatrix();
    Eigen::SparseMatrix<double> flow = M + t * L;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver2;
    solver.compute(flow);
    if (verbose) cout << "Factorized M + tL" << endl;
    solver2.compute(L);
    if (verbose) cout << "Factorized L" << endl;

    Eigen::VectorXd u = solver.solve(u0);

    Eigen::VectorXd divX = Eigen::VectorXd::Zero(u.size());

    std::vector<glm::vec3> tetXs;
    for (Tet t : tets) {
        std::array<Vector3, 4> vertexPositions = layOutIntrinsicTet(t);
        std::array<double, 4> tetU{u[t.verts[0]], u[t.verts[1]], u[t.verts[2]],
                                   u[t.verts[3]]};
        Vector3 tetGradU = grad(tetU, vertexPositions);
        Vector3 X = tetGradU.normalize();

        tetXs.emplace_back(glm::vec3{X.x, X.y, X.z});

        std::array<double, 4> tetDivX = div(X, vertexPositions);
        for (size_t i = 0; i < 4; ++i) {
            divX[t.verts[i]] += tetDivX[i];
        }
    }

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(divX.size());
    divX -= divX.dot(ones) * ones;
    Eigen::VectorXd phi = solver2.solve(divX);

    std::vector<double> distances(phi.data(), phi.data() + phi.size());
    double minDist = distances[0];
    for (size_t i = 1; i < distances.size(); ++i) {
        minDist = fmin(minDist, distances[i]);
    }
    for (size_t i = 0; i < distances.size(); ++i) {
        distances[i] -= minDist;
        assert(distances[i] >= 0);
    }

    return distances;
}

// return the gradient of function u linearly interpolated over a tetrahedron
// with vertices p[0], ... , p[3]
Vector3 grad(std::array<double, 4> u, std::array<Vector3, 4> p) {
    Vector3 gradU{0, 0, 0};
    double vol = dot(p[3] - p[0], cross(p[1] - p[0], p[2] - p[0])) / 6;

    // The gradient of a function which is 1 at vertex 3 and 0 everywhere else
    // is orthogonal to face 012, and has magnitude 1/h (where h is the height
    // of the tetrahedron). We can compute this vector by taking the area
    // normal of the base face and dividing it by 3 times the tet's volume
    // (since volume is 1/3 base * height)

    // Hand code cases because of orientation problems
    // Face 012
    Vector3 areaNormal = 0.5 * cross(p[1] - p[0], p[2] - p[0]);
    gradU += u[3] * areaNormal / (3 * vol);
    // Face 023
    areaNormal = 0.5 * cross(p[2] - p[0], p[3] - p[0]);
    gradU += u[1] * areaNormal / (3 * vol);
    // Face 031
    areaNormal = 0.5 * cross(p[3] - p[0], p[1] - p[0]);
    gradU += u[2] * areaNormal / (3 * vol);
    // Face 213
    areaNormal = 0.5 * cross(p[1] - p[2], p[3] - p[2]);
    gradU += u[0] * areaNormal / (3 * vol);

    return gradU;
}

// returns the integrated divergence of vector field X associated with
// each vertex p[i] of a tetrahedron
std::array<double, 4> div(Vector3 X, std::array<Vector3, 4> p) {
    std::array<double, 4> divX{0, 0, 0, 0};

    // Since X is constant inside of the tet, the divergence associated
    // with each vertex is the flux of X through the face bits associated
    // with that vertex. So to vertex i, we assign divergence
    // 1/3 * (sum of fluxes through faces incident on i).
    // Note that the flux of a constant vector through a face is just
    // that vector dotted with the area normal.

    // We take the dot product of X with each (inward-facing) face normal.
    // Hand code cases because of orientation problems
    // Face 012
    Vector3 areaNormal3 = -0.5 * cross(p[1] - p[0], p[2] - p[0]);
    double flux3        = dot(X, areaNormal3);
    divX[0] += 1 / 3. * flux3;
    divX[1] += 1 / 3. * flux3;
    divX[2] += 1 / 3. * flux3;

    // Face 023
    Vector3 areaNormal1 = -0.5 * cross(p[2] - p[0], p[3] - p[0]);
    double flux1        = dot(X, areaNormal1);
    divX[0] += 1 / 3. * flux1;
    divX[2] += 1 / 3. * flux1;
    divX[3] += 1 / 3. * flux1;

    // Face 031
    Vector3 areaNormal2 = -0.5 * cross(p[3] - p[0], p[1] - p[0]);
    double flux2        = dot(X, areaNormal2);
    divX[0] += 1 / 3. * flux2;
    divX[3] += 1 / 3. * flux2;
    divX[1] += 1 / 3. * flux2;

    // Face 213
    Vector3 areaNormal0 = -0.5 * cross(p[1] - p[2], p[3] - p[2]);
    double flux0        = dot(X, areaNormal0);
    divX[2] += 1 / 3. * flux0;
    divX[1] += 1 / 3. * flux0;
    divX[3] += 1 / 3. * flux0;

    return divX;
}

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
std::array<Vector3, 4> TetMesh::layOutIntrinsicTet(Tet t) {

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

    Vector3 pos0 = Vector3{0, 0, 0};
    Vector3 pos1 = Vector3{e01, 0, 0};
    double angle = cornerAngle(e12, e01, e02);
    Vector3 pos2 = e02 * Vector3{cos(angle), -sin(angle), 0};

    // To place vertex 3, we find the angle at the corner of v0 in face v0-v1-v3
    // Together with the dihedral angle between the 013 face and the 012 face
    // (i.e. the dihedral angle along edge e01), this tells us where v3 should
    // go
    angle                = cornerAngle(e13, e01, e03);
    double dihedralAngle = dihedralAngles(t)[0];
    Vector3 pos3 = e03 * Vector3{cos(angle), -cos(dihedralAngle) * sin(angle),
                                 -sin(dihedralAngle) * sin(angle)};

    std::array<Vector3, 4> positions{pos0, pos1, pos2, pos3};
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
std::array<double, 6> TetMesh::dihedralAngles(Tet t) {
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

    std::array<double, 6> angles{acos(cos01), acos(cos02), acos(cos03),
                                 acos(cos12), acos(cos13), acos(cos23)};

    return angles;
}

double cot(double theta) { return 1 / tan(theta); }

std::array<double, 6> TetMesh::cotanWeights(Tet t) {
    std::array<double, 6> angles = dihedralAngles(t);

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

    // Weird order since the cotan weight for edge e_ij comes from the
    // _opposite_ edge e_kl
    std::array<double, 6> weights{
        e23 * cot(angles[5]) / 6, e13 * cot(angles[4]) / 6,
        e12 * cot(angles[3]) / 6, e03 * cot(angles[2]) / 6,
        e02 * cot(angles[1]) / 6, e01 * cot(angles[0]) / 6};
    return weights;
}

double TetMesh::meanEdgeLength() {
    cout << "WARNING: meanEdgeLength doesn't handle multiplicity correctly"
         << endl;

    double totalEdgeLength = 0;
    size_t nEdges          = 0;
    for (Tet t : tets) {
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

        totalEdgeLength += e01;
        totalEdgeLength += e02;
        totalEdgeLength += e03;
        totalEdgeLength += e12;
        totalEdgeLength += e13;
        totalEdgeLength += e23;

        nEdges += 6;
    }

    return totalEdgeLength / nEdges;
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

        std::array<double, 6> tetWeights = cotanWeights(tets[iT]);
        partialEdgeCotanWeights.insert(partialEdgeCotanWeights.end(),
                                       tetWeights.begin(), tetWeights.end());
    }


    for (auto f : faceList()) {
        Vector3 p0 = vertices[f[0]].position;
        Vector3 p1 = vertices[f[1]].position;
        Vector3 p2 = vertices[f[2]].position;

        double u0 = scaleFactors[f[0]];
        double u1 = scaleFactors[f[1]];
        double u2 = scaleFactors[f[2]];

        double e01 = norm(p0 - p1) * exp(0.5 * (u0 + u1));
        double e02 = norm(p0 - p2) * exp(0.5 * (u0 + u2));
        double e12 = norm(p1 - p2) * exp(0.5 * (u1 + u2));


        faceAreas.emplace_back(area(e01, e02, e12));
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

  std::vector<std::array<size_t, 3>> TetMesh::faceList() {
    std::vector<std::array<size_t, 3>> faces;
    for (Tet t : tets) {
        faces.emplace_back(
            std::array<size_t, 3>{t.verts[0], t.verts[1], t.verts[2]});
        faces.emplace_back(
            std::array<size_t, 3>{t.verts[0], t.verts[2], t.verts[3]});
        faces.emplace_back(
            std::array<size_t, 3>{t.verts[0], t.verts[3], t.verts[1]});
        faces.emplace_back(
            std::array<size_t, 3>{t.verts[2], t.verts[1], t.verts[3]});
    }

    return faces;
}

std::vector<std::array<size_t, 4>> TetMesh::tetList() {
    std::vector<std::array<size_t, 4>> tetCombinatorics;
    for (Tet t : tets) {
        tetCombinatorics.emplace_back(t.verts);
    }

    return tetCombinatorics;
}

std::vector<int> TetMesh::neighborList() {
  std::vector<int> neighbors;
  for (Tet t : tets) {
    for (int n : t.neigh) {
      neighbors.emplace_back(n);
    }
  }

  return neighbors;
}

TetMesh* TetMesh::construct(const std::vector<Vector3>& positions,
                            const std::vector<std::array<size_t, 4>>& tets,
                            const std::vector<std::array<int, 4>>& neigh) {
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
        t.neigh = neigh[n];

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
            std::swap(t.verts[0], t.verts[1]);
            std::swap(t.neigh[0], t.neigh[1]);
        }


        // Add in PartialEdges
        size_t tIdx = tets.size();
        size_t ct = 0;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = i + 1; j < 4; ++j) {
                size_t eIdx = mesh->edges.size();
                t.edges[ct] = eIdx;
                ++ct;
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

    std::vector<std::array<size_t, 4>> tets;
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
            tets.emplace_back(std::array<size_t, 4>{a - 1, b - 1, c - 1, d - 1});
            assert(a - 1 < verts.size());
            assert(b - 1 < verts.size());
            assert(c - 1 < verts.size());
            assert(d - 1 < verts.size());
            // tets.emplace_back(std::vector<size_t>{a , b , c , d });
            assert(idx == tets.size());
        }

        ele.close();
    }

    std::vector<std::array<int, 4>> neighbors;
    ifstream neigh(neighPath);
    if (neigh.is_open()) {
        string line;
        getline(neigh, line);
        while (getline(neigh, line)) {
            if (line[0] == '#' || line.length() == 0) continue;

            istringstream ss(line);
            int idx, a, b, c, d;
            ss >> idx >> a >> b >> c >> d;
            neighbors.emplace_back(std::array<int, 4>{a, b, c, d});
            assert(idx == (int) neighbors.size());
        }

        neigh.close();
    }

    return TetMesh::construct(verts, tets, neighbors);
}
} // namespace CompArch
