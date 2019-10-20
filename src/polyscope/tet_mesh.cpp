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

        faces.emplace_back(std::vector<size_t>{t[0], t[1], t[2]});
        faces.emplace_back(std::vector<size_t>{t[0], t[2], t[3]});
        faces.emplace_back(std::vector<size_t>{t[0], t[3], t[1]});
        faces.emplace_back(std::vector<size_t>{t[2], t[1], t[3]});
    }
    computeGeometryData();

    baseColor = getNextUniqueColor();
    tetColor  = baseColor;
    edgeColor = glm::vec3{0, 0, 0};
}

void TetMesh::computeGeometryData() {
    const glm::vec3 zero{0., 0., 0.};

    // Reset face-valued
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

size_t TetMesh::nFaces() { return faces.size(); }

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
    for (auto& x : quantities) {
        x.second->draw();
    }
    quantitiesMustRefillBuffers = false;

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

void TetMesh::drawPick() {}

void TetMesh::prepare() {
    program.reset(new gl::GLProgram(&gl::PLAIN_SURFACE_VERT_SHADER,
                                    &gl::PLAIN_SURFACE_FRAG_SHADER,
                                    gl::DrawMode::Triangles));

    // Populate draw buffers
    fillGeometryBuffers(*program);

    setMaterialForProgram(*program, "wax");
}

void TetMesh::prepareWireframe() {
    wireframeProgram.reset(new gl::GLProgram(&gl::SURFACE_WIREFRAME_VERT_SHADER,
                                             &gl::SURFACE_WIREFRAME_FRAG_SHADER,
                                             gl::DrawMode::Triangles));

    // Populate draw buffers
    fillGeometryBuffersWireframe(*wireframeProgram);

    setMaterialForProgram(*wireframeProgram, "wax");
}

void TetMesh::preparePick() {}


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

    for (size_t iT = 0; iT < tets.size(); iT++) {
        if (glm::dot(tetCenters[iT], sliceNormal) > sliceDist) continue;
        for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            auto& face = faces[iF];
            assert(face.size() == 3);
            glm::vec3 faceN = faceNormals[iF];

            // implicitly triangulate from root
            size_t vA    = face[0];
            size_t vB    = face[1];
            size_t vC    = face[2];
            glm::vec3 pA = vertices[vA];
            glm::vec3 pB = vertices[vB];
            glm::vec3 pC = vertices[vC];

            positions.push_back(pA);
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

    for (size_t iT = 0; iT < tets.size(); iT++) {
        if (glm::dot(tetCenters[iT], sliceNormal) > sliceDist) continue;
        for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            auto& face      = faces[iF];
            size_t D        = face.size();
            glm::vec3 faceN = faceNormals[iF];

            // implicitly triangulate from root
            size_t vRoot    = face[0];
            glm::vec3 pRoot = vertices[vRoot];
            for (size_t j = 1; (j + 1) < D; j++) {
                size_t vB    = face[j];
                glm::vec3 pB = vertices[vB];
                size_t vC    = face[(j + 1) % D];
                glm::vec3 pC = vertices[vC];

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
    }


    // Store data in buffers
    p.setAttribute("a_position", positions);
    p.setAttribute("a_normal", normals);
    p.setAttribute("a_barycoord", bcoord);
    p.setAttribute("a_edgeReal", edgeReal);
}

// == Build the ImGUI ui elements
void TetMesh::refillBuffers() {
    fillGeometryBuffers(*program);

    if (edgeWidth > 0) fillGeometryBuffersWireframe(*wireframeProgram);

    quantitiesMustRefillBuffers = true;
}

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
        if (ImGui::SliderFloat("Plane Distance", &sliceDist, -10., 10., "%.5f",
                               2.)) {
            refillBuffers();
        }
        if (ImGui::SliderFloat("Plane theta", &sliceTheta, 0., 2. * PI, "%.5f",
                               2.)) {
            sliceNormal =
                glm::vec3{cos(sliceTheta) * cos(slicePhi), sin(slicePhi),
                          sin(sliceTheta) * cos(slicePhi)};
            refillBuffers();
        }
        if (ImGui::SliderFloat("Plane phi", &slicePhi, 0., 2. * PI, "%.5f",
                               2.)) {
            sliceNormal =
                glm::vec3{cos(sliceTheta) * cos(slicePhi), sin(slicePhi),
                          sin(sliceTheta) * cos(slicePhi)};
            refillBuffers();
        }
        ImGui::PopItemWidth();
    }
}
void TetMesh::buildCustomOptionsUI(){}; // overridden by childen to add to the
                                        // options menu
void TetMesh::buildPickUI(
    size_t localPickID){}; // Draw pick UI elements when index localPickID is
                           // selected

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

TetVertexScalarQuantity* TetMesh::addVertexScalarQuantityImpl(
    std::string name, const std::vector<double>& data, DataType type) {
    TetVertexScalarQuantity* q =
        new TetVertexScalarQuantity(name, data, *this, type);
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

} // namespace polyscope
