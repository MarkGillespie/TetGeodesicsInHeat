#include "tet_vector_quantity.h"

#include "polyscope/file_helpers.h"
#include "polyscope/gl/materials/materials.h"
#include "polyscope/gl/shaders.h"
#include "polyscope/gl/shaders/vector_shaders.h"
#include "polyscope/polyscope.h"
#include "polyscope/trace_vector_field.h"

#include "imgui.h"

#include <complex>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

namespace polyscope {

TetVectorQuantity::TetVectorQuantity(std::string name, TetMesh& mesh_,
                                     MeshElement definedOn_,
                                     VectorType vectorType_)
    : TetMeshQuantity(name, mesh_), vectorType(vectorType_),
      definedOn(definedOn_) {}

void TetVectorQuantity::prepareVectorMapper() {

    // Create a mapper (default mapper is identity)
    if (vectorType == VectorType::AMBIENT) {
        mapper.setMinMax(vectors);
    } else {
        mapper = AffineRemapper<glm::vec3>(vectors, DataType::MAGNITUDE);
    }

    // Default viz settings
    if (vectorType != VectorType::AMBIENT) {
        lengthMult = .02;
    } else {
        lengthMult = 1.0;
    }
    radiusMult  = .0005;
    vectorColor = getNextUniqueColor();
}

void TetVectorQuantity::draw() {
    if (!enabled) return;

    if (program == nullptr) {
        prepareProgram();
    }

    // Set uniforms
    parent.setTransformUniforms(*program);

    program->setUniform("u_radius", radiusMult * state::lengthScale);
    program->setUniform("u_color", vectorColor);

    if (vectorType == VectorType::AMBIENT) {
        program->setUniform("u_lengthMult", 1.0);
    } else {
        program->setUniform("u_lengthMult", lengthMult * state::lengthScale);
    }

    program->draw();
}

void TetVectorQuantity::geometryChanged() { program.reset(); }

void TetVectorQuantity::prepareProgram() {

    program.reset(new gl::GLProgram(
        &gl::PASSTHRU_VECTOR_VERT_SHADER, &gl::VECTOR_GEOM_SHADER,
        &gl::SHINY_VECTOR_FRAG_SHADER, gl::DrawMode::Points));

    // Fill buffers
    std::vector<glm::vec3> mappedVectors, mappedRoots;
    for (size_t iV = 0; iV < vectors.size(); ++iV) {
        if (!hideWithMesh || !skipVector(iV)) {
            mappedVectors.push_back(mapper.map(vectors[iV]));
            mappedRoots.push_back(vectorRoots[iV]);
        }
    }

    program->setAttribute("a_vector", mappedVectors);
    program->setAttribute("a_position", mappedRoots);

    setMaterialForProgram(*program, "wax");
}

bool TetVertexVectorQuantity::skipVector(size_t iV) {
    glm::vec3 pos = parent.vertices[iV];
    return glm::dot(pos, parent.sliceNormal) > parent.sliceDist;
}

bool TetFaceVectorQuantity::skipVector(size_t iF) {
    size_t iT     = iF / 4;
    glm::vec3 pos = parent.tetCenters[iT];
    return glm::dot(pos, parent.sliceNormal) > parent.sliceDist;
}

bool TetTetVectorQuantity::skipVector(size_t iT) {
    glm::vec3 pos = parent.tetCenters[iT];
    return glm::dot(pos, parent.sliceNormal) > parent.sliceDist;
}


void TetVectorQuantity::buildCustomUI() {
    ImGui::SameLine();
    ImGui::ColorEdit3("Color", (float*)&vectorColor,
                      ImGuiColorEditFlags_NoInputs);
    ImGui::SameLine();


    // === Options popup
    if (ImGui::Button("Options")) {
        ImGui::OpenPopup("OptionsPopup");
    }
    if (ImGui::BeginPopup("OptionsPopup")) {
        if (ImGui::MenuItem("Write to file")) writeToFile();
        ImGui::EndPopup();
    }


    // Only get to set length for non-ambient vectors
    if (vectorType != VectorType::AMBIENT) {
        ImGui::SliderFloat("Length", &lengthMult, 0.0, 1., "%.5f", 3.);
    }

    ImGui::SliderFloat("Radius", &radiusMult, 0.0, .1, "%.5f", 3.);

    { // Draw max and min magnitude
        ImGui::TextUnformatted(mapper.printBounds().c_str());
    }

    if (ImGui::Checkbox("hide with mesh", &hideWithMesh)) {
        geometryChanged();
    }

    drawSubUI();
}

void TetVectorQuantity::drawSubUI() {}

void TetVectorQuantity::writeToFile(std::string filename) {

    if (filename == "") {
        filename = promptForFilename();
        if (filename == "") {
            return;
        }
    }

    if (options::verbosity > 0) {
        cout << "Writing tet vector quantity " << name << " to file "
             << filename << endl;
    }

    std::ofstream outFile(filename);
    outFile << "#Vectors written by polyscope from Tet Vector Quantity " << name
            << endl;
    outFile << "#displayradius " << (radiusMult * state::lengthScale) << endl;
    outFile << "#displaylength " << (lengthMult * state::lengthScale) << endl;

    for (size_t i = 0; i < vectors.size(); i++) {
        if (glm::length(vectors[i]) > 0) {
            outFile << vectorRoots[i] << " " << vectors[i] << endl;
        }
    }

    outFile.close();
}

// ========================================================
// ==========           Vertex Vector            ==========
// ========================================================

TetVertexVectorQuantity::TetVertexVectorQuantity(
    std::string name, std::vector<glm::vec3> vectors_, TetMesh& mesh_,
    VectorType vectorType_)

    : TetVectorQuantity(name, mesh_, MeshElement::VERTEX, vectorType_),
      vectorField(vectors_) {

    size_t i    = 0;
    vectorRoots = parent.vertices;
    vectors     = vectorField;

    prepareVectorMapper();
}

void TetVertexVectorQuantity::buildVertexInfoGUI(size_t iV) {
    ImGui::TextUnformatted(name.c_str());
    ImGui::NextColumn();

    std::stringstream buffer;
    buffer << vectorField[iV];
    ImGui::TextUnformatted(buffer.str().c_str());

    ImGui::NextColumn();
    ImGui::NextColumn();
    ImGui::Text("magnitude: %g", glm::length(vectorField[iV]));
    ImGui::NextColumn();
}

std::string TetVertexVectorQuantity::niceName() {
    return name + " (vertex vector)";
}

// ========================================================
// ==========            Face Vector             ==========
// ========================================================

TetFaceVectorQuantity::TetFaceVectorQuantity(std::string name,
                                             std::vector<glm::vec3> vectors_,
                                             TetMesh& mesh_,
                                             VectorType vectorType_)
    : TetVectorQuantity(name, mesh_, MeshElement::FACE, vectorType_),
      vectorField(vectors_) {

    // Copy the vectors
    vectors = vectorField;
    vectorRoots.resize(parent.nFaces());
    for (size_t iF = 0; iF < parent.nFaces(); iF++) {
        auto& face           = parent.faces[iF];
        glm::vec3 faceCenter = parent.faceCenter(iF);
        vectorRoots[iF]      = faceCenter;
    }

    prepareVectorMapper();
}

void TetFaceVectorQuantity::buildFaceInfoGUI(size_t iF) {
    ImGui::TextUnformatted(name.c_str());
    ImGui::NextColumn();

    std::stringstream buffer;
    buffer << vectorField[iF];
    ImGui::TextUnformatted(buffer.str().c_str());

    ImGui::NextColumn();
    ImGui::NextColumn();
    ImGui::Text("magnitude: %g", glm::length(vectorField[iF]));
    ImGui::NextColumn();
}

std::string TetFaceVectorQuantity::niceName() {
    return name + " (face vector)";
}

// ========================================================
// ==========            Tet Vector             ==========
// ========================================================

TetTetVectorQuantity::TetTetVectorQuantity(std::string name,
                                           std::vector<glm::vec3> vectors_,
                                           TetMesh& mesh_,
                                           VectorType vectorType_)
    // TODO: fix enum to remove hack
    : TetVectorQuantity(name, mesh_, MeshElement::CORNER, vectorType_),
      vectorField(vectors_) {

    // Copy the vectors
    vectors = vectorField;
    vectorRoots.resize(parent.nTets());
    for (size_t iT = 0; iT < parent.nTets(); iT++) {
        auto& tet           = parent.tets[iT];
        glm::vec3 tetCenter = parent.tetCenters[iT];
        vectorRoots[iT]     = tetCenter;
    }

    prepareVectorMapper();
}

void TetTetVectorQuantity::buildTetInfoGUI(size_t iT) {
    ImGui::TextUnformatted(name.c_str());
    ImGui::NextColumn();

    std::stringstream buffer;
    buffer << vectorField[iT];
    ImGui::TextUnformatted(buffer.str().c_str());

    ImGui::NextColumn();
    ImGui::NextColumn();
    ImGui::Text("magnitude: %g", glm::length(vectorField[iT]));
    ImGui::NextColumn();
}

std::string TetTetVectorQuantity::niceName() { return name + " (tet vector)"; }
} // namespace polyscope
