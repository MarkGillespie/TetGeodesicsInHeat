#include "tet_mesh.h"

namespace polyscope {

const std::string TetMesh::structureTypeName = "Tet Mesh";
std::string TetMesh::typeName() { return structureTypeName; }

TetMesh::TetMesh(std::string name_)
    : QuantityStructure<TetMesh>(name_), name(name_) {}

TetMesh::TetMesh(std::string name_, std::vector<glm::vec3> vertices_,
                 std::vector<std::vector<size_t>> tets_)
    : QuantityStructure<TetMesh>(name_), vertices(vertices_), tets(tets_),
      name(name_) {

    for (size_t i = 0; i < tets.size(); ++i) {
        std::vector<size_t> t = tets[i];

        // In order OppV0, OppV1, OppV2, OppV3
        localFaces.emplace_back(std::array<size_t, 3>{2, 1, 3});
        localFaces.emplace_back(std::array<size_t, 3>{0, 2, 3});
        localFaces.emplace_back(std::array<size_t, 3>{0, 3, 1});
        localFaces.emplace_back(std::array<size_t, 3>{0, 1, 2});

        faces.emplace_back(std::vector<size_t>{t[2], t[1], t[3]});
        faces.emplace_back(std::vector<size_t>{t[0], t[2], t[3]});
        faces.emplace_back(std::vector<size_t>{t[0], t[3], t[1]});
        faces.emplace_back(std::vector<size_t>{t[0], t[1], t[2]});
    }

    computeGeometryData();

    baseColor = getNextUniqueColor();
    tetColor  = baseColor;
    edgeColor = glm::vec3{0, 0, 0};
}

void TetMesh::computeGeometryData() {
    const glm::vec3 zero{0., 0., 0.};

    // Reset face-valued
    faceAreas.resize(faces.size());
    faceNormals.resize(faces.size());

    // Loop over faces to compute face-valued quantities
    for (size_t iF = 0; iF < faces.size(); iF++) {
        auto& face = faces[iF];
        assert(face.size() == 3);

        glm::vec3 fN = zero;
        double fA    = 0;
        glm::vec3 pA = vertices[face[0]];
        glm::vec3 pB = vertices[face[1]];
        glm::vec3 pC = vertices[face[2]];

        fN = glm::cross(pB - pA, pC - pA);
        fA = 0.5 * glm::length(fN);

        // Set face values
        fN              = glm::normalize(fN);
        faceNormals[iF] = fN;
        faceAreas[iF]   = fA;
    }

    // Loop over tets to compute centers and vertex areas
    vertexVolumes = std::vector<double>(vertices.size(), 0.0);
    for (size_t iT = 0; iT < tets.size(); ++iT) {
        glm::vec3 p0     = vertices[tets[iT][0]];
        glm::vec3 p1     = vertices[tets[iT][1]];
        glm::vec3 p2     = vertices[tets[iT][2]];
        glm::vec3 p3     = vertices[tets[iT][3]];
        glm::vec3 center = (p0 + p1 + p2 + p3) / 4.f;
        tetCenters.emplace_back(center);

        double vol = glm::dot(p0 - p3, glm::cross(p1 - p3, p2 - p3)) / 6;
        for (size_t i = 0; i < 4; ++i) {
            vertexVolumes[tets[iT][i]] += vol / 4;
        }
    }
}


glm::vec3 TetMesh::faceCenter(size_t iF) {
    std::vector<size_t> f = faces[iF];
    glm::vec3 center      = vertices[f[0]] + vertices[f[1]] + vertices[f[2]];
    return center / 3.f;
}

size_t TetMesh::nVertices() { return vertices.size(); }
size_t TetMesh::nFaces() { return faces.size(); }
size_t TetMesh::nTets() { return tets.size(); }

void TetMesh::draw() {
    if (!enabled) {
        return;
    }

    if (dominantQuantity == nullptr) {
        if (program == nullptr) {
            prepare();
        }

        // Set uniforms
        setTransformUniforms(*program);
        program->setUniform("u_basecolor", tetColor);

        program->draw();
    }

    // Draw the quantities
    // TODO: draw quantities
    for (auto& x : quantities) {
        // x.second->draw();
    }

    // Draw the wireframe
    if (edgeWidth > 0) {
        if (wireframeProgram == nullptr) {
            prepareWireframe();
        }

        // Set uniforms
        setTransformUniforms(*wireframeProgram);
        wireframeProgram->setUniform("u_edgeWidth", edgeWidth);
        wireframeProgram->setUniform("u_edgeColor", edgeColor);

        glEnable(GL_BLEND);
        glDepthFunc(GL_LEQUAL); // Make sure wireframe wins depth tests
        glBlendFuncSeparate(
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO,
            GL_ONE); // slightly weird blend function: ensures alpha is set by
                     // whatever was drawn before, rather than the wireframe

        wireframeProgram->draw();

        glDepthFunc(GL_LESS); // return to normal
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
}

void TetMesh::drawPick() {
    if (!enabled) {
        return;
    }

    if (pickProgram == nullptr) {
        preparePick();
    }

    // Set uniforms
    setTransformUniforms(*pickProgram);

    pickProgram->draw();
}

void TetMesh::prepare() {
    program.reset(new gl::GLProgram(&gl::PLAIN_SURFACE_VERT_SHADER,
                                    &gl::PLAIN_SURFACE_FRAG_SHADER,
                                    gl::DrawMode::Triangles));

    if (!bufferGeometryValid) precomputeBufferGeometry();

    // Populate draw buffers
    fillGeometryBuffers(*program);

    setMaterialForProgram(*program, "wax");
}

void TetMesh::prepareWireframe() {
    wireframeProgram.reset(new gl::GLProgram(&gl::SURFACE_WIREFRAME_VERT_SHADER,
                                             &gl::SURFACE_WIREFRAME_FRAG_SHADER,
                                             gl::DrawMode::Triangles));

    if (!bufferGeometryValid) precomputeBufferGeometry();

    // Populate draw buffers
    fillGeometryBuffersWireframe(*wireframeProgram);

    setMaterialForProgram(*wireframeProgram, "wax");
}

void TetMesh::preparePick() {

    // Create a new program
    pickProgram.reset(new gl::GLProgram(&gl::PICK_SURFACE_VERT_SHADER,
                                        &gl::PICK_SURFACE_FRAG_SHADER,
                                        gl::DrawMode::Triangles));

    if (!bufferGeometryValid) precomputeBufferGeometry();
    fillGeometryBuffersPick(*pickProgram);
}

void TetMesh::precomputeBufferGeometry() {
    bufferVertices.clear();
    bufferFaces.clear();
    bufferFaceNormals.clear();
    bufferVertices.reserve(3 * faces.size());
    bufferFaces.reserve(3 * faces.size());
    bufferFaceNormals.reserve(3 * faces.size());

    if (sliceThroughTets) {
        for (size_t iT = 0; iT < tets.size(); ++iT) {
            std::array<double, 4> fn;
            for (size_t iV = 0; iV < 4; ++iV) {
                fn[iV] = sliceDist - dot(vertices[tets[iT][iV]], sliceNormal);
            }
            sliceTet(iT, fn, bufferVertices, bufferFaces, bufferFaceNormals);
        }
    } else {
        for (size_t iT = 0; iT < tets.size(); ++iT) {
            if (glm::dot(tetCenters[iT], sliceNormal) <= sliceDist) {
                bufferVertices = vertices;
                for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
                    bufferFaces.emplace_back(faces[iF]);
                    bufferFaceNormals.emplace_back(faceNormals[iF]);
                }
            }
        }
    }
}

// if v does not contain i, return false
// if v contains i, return true, and cyclically shift v so that i is at the end
bool contains(std::array<size_t, 3>& v, size_t i) {
    if (v[0] == i) {
        v[0] = v[1];
        v[1] = v[2];
        v[2] = i;
        return true;
    } else if (v[1] == i) {
        v[1] = v[0];
        v[0] = v[2];
        v[2] = i;
        return true;
    } else if (v[2] == i) {
        return true;
    }
    return false;
}

// Computes the region inside of tet iT where the function fn is nonnegative
// (assuming you linearly interpolate fn inside of the tet)
void TetMesh::sliceTet(size_t iT, std::array<double, 4> fn,
                       std::vector<glm::vec3>& vs,
                       std::vector<std::vector<size_t>>& fs,
                       std::vector<glm::vec3>& Ns) {
    std::array<std::tuple<double, size_t>, 4> corners;
    for (size_t i = 0; i < 4; ++i) corners[i] = std::make_tuple(fn[i], i);

    // Now all of the negative vertices are at the beginning and all of the
    // positive vertices are at the end
    std::sort(corners.begin(), corners.end());

    std::array<size_t, 4> s2l;
    for (size_t i = 0; i < 4; ++i) s2l[i] = std::get<1>(corners[i]);

    auto intersect = [&](size_t a, size_t b) {
        float t = (0 - fn[a]) / (fn[b] - fn[a]);
        return (1.f - t) * vertices[tets[iT][a]] + t * vertices[tets[iT][b]];
    };

    // TODO: write this better using the observation that faces are opposite
    // vertices (and thus you can tell which normal to use just from the
    // vertices contained in the face)
    size_t baseV = vs.size();
    if (fn[s2l[3]] < 0) {
        // Negative everywhere - return nothing
    } else if (fn[s2l[2]] < 0) {
        // Negative everywhere except vertex 3. Return a pyramid with tip at
        // vertex 3

        size_t real  = s2l[3];
        size_t imag1 = s2l[0];
        size_t imag2 = s2l[1];
        size_t imag3 = s2l[2];

        vs.emplace_back(vertices[tets[iT][real]]);
        vs.emplace_back(intersect(real, imag1));
        vs.emplace_back(intersect(real, imag2));
        vs.emplace_back(intersect(real, imag3));

        // Real 1, Imag 1, Imag 2 (missing imag 3)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV, baseV + 1, baseV + 2});
        Ns.emplace_back(faceNormals[tets[iT][imag3]]);

        // Real 1, Imag 1, Imag 3 (missing imag 2)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV, baseV + 1, baseV + 3});
        Ns.emplace_back(faceNormals[tets[iT][imag2]]);

        // Real 1, Imag 2, Imag 3 (missing Imag 1)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV, baseV + 2, baseV + 3});
        Ns.emplace_back(faceNormals[tets[iT][imag1]]);

        // All Imag (missing real)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV + 1, baseV + 2, baseV + 3});
        // Normal vector should point away from barycenter
        glm::vec3 v12       = vs[baseV + 2] - vs[baseV + 1];
        glm::vec3 v13       = vs[baseV + 3] - vs[baseV + 1];
        glm::vec3 newNormal = glm::cross(v12, v13);
        glm::vec3 tetBarycenter =
            0.25f * (vs[baseV] + vs[baseV + 1] + vs[baseV + 2] + vs[baseV + 3]);
        glm::vec3 faceBarycenter =
            1.f / 3.f * (vs[baseV + 1] + vs[baseV + 2] + vs[baseV + 3]);
        if (glm::dot(newNormal, faceBarycenter - tetBarycenter) < 0) {
            newNormal *= -1;
        }
        Ns.emplace_back(glm::normalize(newNormal));

    } else if (fn[s2l[1]] < 0) {
        // Cut in half
        size_t real1 = s2l[2];
        size_t real2 = s2l[3];
        size_t imag1 = s2l[0];
        size_t imag2 = s2l[1];

        vs.emplace_back(vertices[tets[iT][real1]]);
        vs.emplace_back(intersect(real1, imag1));
        vs.emplace_back(intersect(real1, imag2));
        vs.emplace_back(vertices[tets[iT][real2]]);
        vs.emplace_back(intersect(real2, imag1));
        vs.emplace_back(intersect(real2, imag2));

        // Real 1, Real 2, Imag 1 (missing imag 2)
        fs.emplace_back((std::initializer_list<size_t>){baseV, baseV + 3,
                                                        baseV + 4, baseV + 1});
        Ns.emplace_back(faceNormals[tets[iT][imag2]]);

        // Real 1, Real 2, Imag 2 (missing imag 1)
        fs.emplace_back((std::initializer_list<size_t>){baseV, baseV + 3,
                                                        baseV + 5, baseV + 2});
        Ns.emplace_back(faceNormals[tets[iT][imag1]]);

        // Real 1, Imag 1, Imag 2 (missing Real 2)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV, baseV + 1, baseV + 2});
        Ns.emplace_back(faceNormals[tets[iT][real2]]);

        // Real 2, Imag 1, Imag 2 (missing Real 1)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV + 3, baseV + 4, baseV + 5});
        Ns.emplace_back(faceNormals[tets[iT][real1]]);

        // All imag (average real normals?)
        fs.emplace_back(
            std::vector<size_t>{baseV + 1, baseV + 2, baseV + 5, baseV + 4});
        glm::vec3 avgRealNormal = 0.5f * (faceNormals[tets[iT][real1]] +
                                          faceNormals[tets[iT][real2]]);

        // Normal vector should point away from barycenter
        glm::vec3 v12       = vs[baseV + 2] - vs[baseV + 1];
        glm::vec3 v15       = vs[baseV + 5] - vs[baseV + 1];
        glm::vec3 newNormal = glm::cross(v12, v15);
        glm::vec3 tetBarycenter =
            1.f / 6.f *
            (vs[baseV] + vs[baseV + 1] + vs[baseV + 2] + vs[baseV + 3] +
             vs[baseV + 4] + vs[baseV + 5]);
        glm::vec3 faceBarycenter =
            1.f / 4.f *
            (vs[baseV + 1] + vs[baseV + 2] + vs[baseV + 4] + vs[baseV + 5]);
        if (glm::dot(newNormal, faceBarycenter - tetBarycenter) < 0) {
            newNormal *= -1;
        }
        Ns.emplace_back(glm::normalize(newNormal));

    } else if (fn[s2l[0]] < 0) {
        // Positive everywhere except vertex 0. Return the complement of a
        // pyramid at vertex 0
        size_t real1 = s2l[1];
        size_t real2 = s2l[2];
        size_t real3 = s2l[3];
        size_t imag  = s2l[0];

        vs.emplace_back(vertices[tets[iT][real1]]);
        vs.emplace_back(vertices[tets[iT][real2]]);
        vs.emplace_back(vertices[tets[iT][real3]]);
        vs.emplace_back(intersect(real1, imag));
        vs.emplace_back(intersect(real2, imag));
        vs.emplace_back(intersect(real3, imag));

        // Real 1, Real 2, Imag (missing Real 3)
        fs.emplace_back((std::initializer_list<size_t>){baseV, baseV + 1,
                                                        baseV + 4, baseV + 3});
        Ns.emplace_back(faceNormals[tets[iT][real3]]);
        // Real 1, Real 3, Imag (missing Real 2)
        fs.emplace_back((std::initializer_list<size_t>){baseV, baseV + 2,
                                                        baseV + 5, baseV + 3});
        Ns.emplace_back(faceNormals[tets[iT][real2]]);
        // Real 2, Real 3, Imag (missing Real 1)
        fs.emplace_back((std::initializer_list<size_t>){baseV + 1, baseV + 2,
                                                        baseV + 5, baseV + 4});
        Ns.emplace_back(faceNormals[tets[iT][real1]]);
        // Real 1, Real 2, Real 3, (missing Imag)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV, baseV + 1, baseV + 2});
        Ns.emplace_back(faceNormals[tets[iT][imag]]);
        // Vertex cap (opposite face missing imag)
        fs.emplace_back(
            (std::initializer_list<size_t>){baseV + 3, baseV + 4, baseV + 5});

        // Normal vector should point away from barycenter
        glm::vec3 v34       = vs[baseV + 5] - vs[baseV + 3];
        glm::vec3 v35       = vs[baseV + 4] - vs[baseV + 3];
        glm::vec3 newNormal = glm::cross(v34, v35);
        glm::vec3 tetBarycenter =
            1.f / 6.f *
            (vs[baseV] + vs[baseV + 1] + vs[baseV + 2] + vs[baseV + 3] +
             vs[baseV + 4] + vs[baseV + 5]);
        glm::vec3 faceBarycenter =
            1.f / 3.f * (vs[baseV + 3] + vs[baseV + 4] + vs[baseV + 5]);
        if (glm::dot(newNormal, faceBarycenter - tetBarycenter) < 0) {
            newNormal *= -1;
        }
        Ns.emplace_back(glm::normalize(newNormal));
    } else {
        // Positive everywhere - return everything
        for (size_t iV : tets[iT]) {
            vs.emplace_back(vertices[iV]);
        }
        for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            std::vector<size_t> f;
            for (size_t iV : localFaces[iF]) {
                f.emplace_back(iV + baseV);
            }
            fs.emplace_back(f);
            Ns.emplace_back(faceNormals[iF]);
        }
    }
}

void TetMesh::fillGeometryBuffers(gl::GLProgram& p) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> bcoord;

    bool wantsBary = p.hasAttribute("a_barycoord");

    positions.reserve(3 * faces.size());
    normals.reserve(3 * faces.size());
    if (wantsBary) {
        bcoord.reserve(3 * faces.size());
    }

    for (size_t iF = 0; iF < bufferFaces.size(); iF++) {
        auto& face      = bufferFaces[iF];
        size_t D        = face.size();
        glm::vec3 faceN = bufferFaceNormals[iF];

        // implicitly triangulate from root
        size_t vRoot    = face[0];
        glm::vec3 pRoot = bufferVertices[vRoot];
        for (size_t j = 1; (j + 1) < D; j++) {
            size_t vB    = face[j];
            glm::vec3 pB = bufferVertices[vB];
            size_t vC    = face[(j + 1) % D];
            glm::vec3 pC = bufferVertices[vC];

            positions.push_back(pRoot);
            positions.push_back(pB);
            positions.push_back(pC);

            normals.push_back(faceN);
            normals.push_back(faceN);
            normals.push_back(faceN);

            if (wantsBary) {
                bcoord.push_back(glm::vec3{1., 0., 0.});
                bcoord.push_back(glm::vec3{0., 1., 0.});
                bcoord.push_back(glm::vec3{0., 0., 1.});
            }
        }
    }

    // Store data in buffers
    p.setAttribute("a_position", positions);
    p.setAttribute("a_normal", normals);
    if (wantsBary) {
        p.setAttribute("a_barycoord", bcoord);
    }
}

void TetMesh::fillGeometryBuffersWireframe(gl::GLProgram& p) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> bcoord;
    std::vector<glm::vec3> edgeReal;

    positions.reserve(3 * faces.size());
    normals.reserve(3 * faces.size());
    bcoord.reserve(3 * faces.size());
    edgeReal.reserve(3 * faces.size());

    for (size_t iF = 0; iF < bufferFaces.size(); iF++) {
        auto& face      = bufferFaces[iF];
        size_t D        = face.size();
        glm::vec3 faceN = bufferFaceNormals[iF];

        // implicitly triangulate from root
        size_t vRoot    = face[0];
        glm::vec3 pRoot = bufferVertices[vRoot];
        for (size_t j = 1; (j + 1) < D; j++) {
            size_t vB    = face[j];
            glm::vec3 pB = bufferVertices[vB];
            size_t vC    = face[(j + 1) % D];
            glm::vec3 pC = bufferVertices[vC];

            positions.push_back(pRoot);
            positions.push_back(pB);
            positions.push_back(pC);

            normals.push_back(faceN);
            normals.push_back(faceN);
            normals.push_back(faceN);

            bcoord.push_back(glm::vec3{1., 0., 0.});
            bcoord.push_back(glm::vec3{0., 1., 0.});
            bcoord.push_back(glm::vec3{0., 0., 1.});

            glm::vec3 edgeRealV{0., 1., 0.};
            if (j == 1) {
                edgeRealV.x = 1.;
            }
            if (j + 2 == D) {
                edgeRealV.z = 1.;
            }
            edgeReal.push_back(edgeRealV);
            edgeReal.push_back(edgeRealV);
            edgeReal.push_back(edgeRealV);
        }
    }

    // Store data in buffers
    p.setAttribute("a_position", positions);
    p.setAttribute("a_normal", normals);
    p.setAttribute("a_barycoord", bcoord);
    p.setAttribute("a_edgeReal", edgeReal);
}

void TetMesh::fillGeometryBuffersPick(gl::GLProgram& p) {
    // Get element indices
    size_t totalPickElements = nVertices() + nFaces();

    // In "local" indices, indexing elements only within this triMesh, used for
    // reading later
    facePickIndStart = nVertices();

    // In "global" indices, indexing all elements in the scene, used to fill
    // buffers for drawing here
    size_t pickStart = pick::requestPickBufferRange(this, totalPickElements);
    size_t faceGlobalPickIndStart = pickStart + nVertices();

    // == Fill buffers
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> bcoord;
    std::vector<std::array<glm::vec3, 3>> vertexColors, edgeColors,
        halfedgeColors;
    std::vector<glm::vec3> faceColor;

    // Reserve space
    positions.reserve(3 * nFaces());
    bcoord.reserve(3 * nFaces());
    vertexColors.reserve(3 * nFaces());
    edgeColors.reserve(3 * nFaces());
    halfedgeColors.reserve(3 * nFaces());
    faceColor.reserve(3 * nFaces());

    // for now, we just color edges and halfedges their respective face's colors
    // TODO: figure out edges

    // TODO: fix indices - everything got broken by clipping
    // Build all quantities in each face
    for (size_t iF = 0; iF < bufferFaces.size(); iF++) {
        auto& face      = bufferFaces[iF];
        size_t D        = face.size();
        glm::vec3 faceN = bufferFaceNormals[iF];

        // implicitly triangulate from root
        size_t vRoot    = face[0];
        glm::vec3 pRoot = bufferVertices[vRoot];
        for (size_t j = 1; (j + 1) < D; j++) {
            size_t vB    = face[j];
            glm::vec3 pB = bufferVertices[vB];
            size_t vC    = face[(j + 1) % D];
            glm::vec3 pC = bufferVertices[vC];

            glm::vec3 fColor = pick::indToVec(iF + faceGlobalPickIndStart);
            std::array<size_t, 3> vertexInds = {vRoot, vB, vC};

            positions.push_back(pRoot);
            positions.push_back(pB);
            positions.push_back(pC);

            // Build all quantities
            std::array<glm::vec3, 3> vColor;
            std::array<glm::vec3, 3> constFColor;

            for (size_t i = 0; i < 3; i++) {
                // Want just one copy of face color, so we can build it in the
                // usual way
                faceColor.push_back(fColor);
                constFColor[i] = fColor;

                // Vertex index color
                vColor[i] = pick::indToVec(vertexInds[i] + pickStart);
            }


            // Push three copies of the values needed at each vertex
            for (int j = 0; j < 3; j++) {
                vertexColors.push_back(vColor);
                edgeColors.push_back(constFColor);
                halfedgeColors.push_back(constFColor);
            }

            // Barycoords
            bcoord.push_back(glm::vec3{1.0, 0.0, 0.0});
            bcoord.push_back(glm::vec3{0.0, 1.0, 0.0});
            bcoord.push_back(glm::vec3{0.0, 0.0, 1.0});
        }
    }

    // Store data in buffers
    pickProgram->setAttribute("a_position", positions);
    pickProgram->setAttribute("a_barycoord", bcoord);
    pickProgram->setAttribute<glm::vec3, 3>("a_vertexColors", vertexColors);
    pickProgram->setAttribute<glm::vec3, 3>("a_edgeColors", edgeColors);
    pickProgram->setAttribute<glm::vec3, 3>("a_halfedgeColors", halfedgeColors);
    pickProgram->setAttribute("a_faceColor", faceColor);
}

// == Build the ImGUI ui elements
void TetMesh::buildCustomUI() {
    // Print stats
    long long int nVertsL = static_cast<long long int>(vertices.size());
    long long int nFacesL = static_cast<long long int>(faces.size());
    long long int nTetsL  = static_cast<long long int>(tets.size());
    ImGui::Text("#verts: %lld  #faces: %lld\n#tets: %lld", nVertsL, nFacesL,
                nTetsL);

    { // colors
        ImGui::ColorEdit3("Color", (float*)&tetColor,
                          ImGuiColorEditFlags_NoInputs);
        ImGui::SameLine();
        ImGui::PushItemWidth(100);
        ImGui::ColorEdit3("Edge Color", (float*)&edgeColor,
                          ImGuiColorEditFlags_NoInputs);
        ImGui::PopItemWidth();
    }

    { // Edge width
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("Edge Width", &edgeWidth, 0.0, 1., "%.5f", 2.);
        if (ImGui::SliderFloat("Plane Distance", &sliceDist, -1., 1., "%.5f",
                               2.)) {
            geometryChanged();
        }
        if (ImGui::SliderFloat("Plane theta", &sliceTheta, 0., 2. * PI, "%.5f",
                               2.)) {
            sliceNormal =
                glm::vec3{cos(sliceTheta) * cos(slicePhi), sin(slicePhi),
                          sin(sliceTheta) * cos(slicePhi)};
            geometryChanged();
        }
        if (ImGui::SliderFloat("Plane phi", &slicePhi, 0., 2. * PI, "%.5f",
                               2.)) {
            sliceNormal =
                glm::vec3{cos(sliceTheta) * cos(slicePhi), sin(slicePhi),
                          sin(sliceTheta) * cos(slicePhi)};
            geometryChanged();
        }
        if (ImGui::Checkbox("Slice Through Tets", &sliceThroughTets)) {
            geometryChanged();
        }
        ImGui::PopItemWidth();
    }
}
void TetMesh::buildCustomOptionsUI(){};

// Draw pick UI elements when index localPickID is
// selected
// TODO: add GUI for quantities
void TetMesh::buildPickUI(size_t localPickID) {
    // Selection type
    if (localPickID < facePickIndStart) {
        buildVertexInfoGui(localPickID);
    } else {
        buildFaceInfoGui(localPickID - facePickIndStart);
    }
};

void TetMesh::buildVertexInfoGui(size_t vInd) {

    size_t displayInd = vInd;
    ImGui::TextUnformatted(("Vertex #" + std::to_string(displayInd)).c_str());

    std::stringstream buffer;
    buffer << vertices[vInd];
    ImGui::TextUnformatted(("Position: " + buffer.str()).c_str());

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Indent(20.);

    // Build GUI to show the quantities
    ImGui::Columns(2);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / 3);
    for (auto& x : quantities) {
        x.second->buildVertexInfoGUI(vInd);
    }

    ImGui::Indent(-20.);
}

void TetMesh::buildFaceInfoGui(size_t fInd) {
    size_t displayInd = fInd;
    ImGui::TextUnformatted(("Face #" + std::to_string(displayInd)).c_str());

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Indent(20.);

    // Build GUI to show the quantities
    ImGui::Columns(2);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / 3);
    for (auto& x : quantities) {
        x.second->buildFaceInfoGUI(fInd);
    }

    ImGui::Indent(-20.);
}


// = Length and bounding box (returned in object coordinates)
std::tuple<glm::vec3, glm::vec3> TetMesh::boundingBox() {
    glm::vec3 min{100, 100, 100};
    glm::vec3 max{-100, -100, -100};
    for (glm::vec3 v : vertices) {
        min.x = fmin(min.x, v.x);
        min.y = fmin(min.y, v.y);
        min.z = fmin(min.z, v.z);
        max.x = fmax(max.x, v.x);
        max.y = fmax(max.y, v.y);
        max.z = fmax(max.z, v.z);
    }
    return std::make_tuple(min, max);
}
double TetMesh::lengthScale() {
    std::tuple<glm::vec3, glm::vec3> bb = boundingBox();
    return glm::distance(std::get<1>(bb), std::get<0>(bb));
}


void TetMesh::geometryChanged() {
    bufferGeometryValid = false;
    program.reset();
    wireframeProgram.reset();
    pickProgram.reset();

    computeGeometryData();

    for (auto& q : quantities) {
        q.second->geometryChanged();
    }
}

TetVertexScalarQuantity* TetMesh::addVertexScalarQuantityImpl(
    std::string name, const std::vector<double>& data, DataType type) {
    TetVertexScalarQuantity* q =
        new TetVertexScalarQuantity(name, data, *this, type);
    addQuantity(q);
    return q;
}

TetFaceScalarQuantity* TetMesh::addFaceScalarQuantityImpl(
    std::string name, const std::vector<double>& data, DataType type) {
    TetFaceScalarQuantity* q =
        new TetFaceScalarQuantity(name, data, *this, type);
    addQuantity(q);
    return q;
}


TetVertexVectorQuantity*
TetMesh::addVertexVectorQuantityImpl(std::string name,
                                     const std::vector<glm::vec3>& vectors,
                                     VectorType vectorType) {
    TetVertexVectorQuantity* q =
        new TetVertexVectorQuantity(name, vectors, *this, vectorType);
    addQuantity(q);
    return q;
}

TetFaceVectorQuantity*
TetMesh::addFaceVectorQuantityImpl(std::string name,
                                   const std::vector<glm::vec3>& vectors,
                                   VectorType vectorType) {

    TetFaceVectorQuantity* q =
        new TetFaceVectorQuantity(name, vectors, *this, vectorType);
    addQuantity(q);
    return q;
}

TetTetVectorQuantity*
TetMesh::addTetVectorQuantityImpl(std::string name,
                                  const std::vector<glm::vec3>& vectors,
                                  VectorType vectorType) {

    TetTetVectorQuantity* q =
        new TetTetVectorQuantity(name, vectors, *this, vectorType);
    addQuantity(q);
    return q;
}

TetMeshQuantity::TetMeshQuantity(std::string name, TetMesh& parentStructure,
                                 bool dominates)
    : Quantity<TetMesh>(name, parentStructure, dominates) {}
void TetMeshQuantity::geometryChanged(){};
void TetMeshQuantity::buildVertexInfoGUI(size_t vInd) {}
void TetMeshQuantity::buildFaceInfoGUI(size_t fInd) {}
void TetMeshQuantity::buildEdgeInfoGUI(size_t eInd) {}
void TetMeshQuantity::buildHalfedgeInfoGUI(size_t heInd) {}
void TetMeshQuantity::buildTetInfoGUI(size_t heInd) {}

} // namespace polyscope
