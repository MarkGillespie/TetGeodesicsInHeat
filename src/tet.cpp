#include "tet.h"

TetMesh::TetMesh(){}

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
        faces.emplace_back(std::vector<size_t>{t.verts[2], t.verts[3], t.verts[1]});
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

    for (size_t n = 0; n < tets.size(); ++n) {
        Tet t;
        t.verts = tets[n];
        // reorder vertices to be positively oriented
        // the tetrahedron is positively oriented if triangle v0, v1, v2 has a normal vector pointing towards v3 
        Vector3 v0 = mesh->vertices[t.verts[0]].position;
        Vector3 v1 = mesh->vertices[t.verts[1]].position;
        Vector3 v2 = mesh->vertices[t.verts[2]].position;
        Vector3 v3 = mesh->vertices[t.verts[3]].position;
        Vector3 normal = cross(v1 - v0, v2 - v0);
        Vector3 toV3 = v3 - (v0 + v1 + v2) / 3;
        if (dot(normal, toV3) < 0) {
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
