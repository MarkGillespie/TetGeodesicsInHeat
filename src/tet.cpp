#include "tet.h"

Tet::Tet() : verts({0, 0, 0, 0}), a(verts[0]), b(verts[1]), c(verts[2]), d(verts[3]), neigh({0, 0, 0, 0}), A(neigh[0]), B(neigh[1]), C(neigh[2]), D(neigh[3]) {}

TetMesh::TetMesh(){}

TetMesh* TetMesh::construct(const std::vector<Vector3>& positions,
            const std::vector<std::vector<size_t>>& tets,
            const std::vector<std::vector<size_t>>& neigh) {
    TetMesh* mesh = new TetMesh();

    for (Vector3 p : positions) {
        Vertex v;
        v.position = p;
        mesh->vertices.emplace_back(v);
    }

    for (size_t i = 0; i < tets.size(); ++i) {
        Tet t;
        t.verts = tets[i];
        t.neigh = neigh[i];
        size_t tIdx = tets.size();

        t.ab = mesh->edges.size();;
        mesh->vertices[t.a].edges.emplace_back(t.ab);
        mesh->vertices[t.b].edges.emplace_back(t.ab);
        mesh->edges.emplace_back(PartialEdge{t.a, t.b, tIdx});

        t.ac = mesh->edges.size();;
        mesh->vertices[t.a].edges.emplace_back(t.ac);
        mesh->vertices[t.c].edges.emplace_back(t.ac);
        mesh->edges.emplace_back(PartialEdge{t.a, t.c, tIdx});

        t.ad = mesh->edges.size();
        mesh->vertices[t.a].edges.emplace_back(t.ad);
        mesh->vertices[t.d].edges.emplace_back(t.ad);
        mesh->edges.emplace_back(PartialEdge{t.a, t.d, tIdx});

        t.bc = mesh->edges.size();
        mesh->vertices[t.b].edges.emplace_back(t.bc);
        mesh->vertices[t.c].edges.emplace_back(t.bc);
        mesh->edges.emplace_back(PartialEdge{t.b, t.c, tIdx});

        t.bd = mesh->edges.size();
        mesh->vertices[t.b].edges.emplace_back(t.bd);
        mesh->vertices[t.d].edges.emplace_back(t.bd);
        mesh->edges.emplace_back(PartialEdge{t.b, t.d, tIdx});

        t.cd = mesh->edges.size();
        mesh->vertices[t.c].edges.emplace_back(t.cd);
        mesh->vertices[t.d].edges.emplace_back(t.cd);
        mesh->edges.emplace_back(PartialEdge{t.c, t.d, tIdx});

        mesh->tets.emplace_back(t);
    }

    for (Tet t : mesh->tets) {
        // opposite tet A shares vertices b, c, d
        for (size_t i = 0; i < mesh->tets.size(); ++i) {

        }
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
            tets.emplace_back(std::vector<size_t>{a, b, c, d});
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
