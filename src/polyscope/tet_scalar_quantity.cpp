#include "tet_scalar_quantity.h"


#include "polyscope/file_helpers.h"
#include "polyscope/gl/materials/materials.h"
#include "polyscope/gl/shaders.h"
#include "polyscope/gl/shaders/surface_shaders.h"
#include "polyscope/polyscope.h"

#include "imgui.h"

using std::cout;
using std::endl;

namespace polyscope {

TetScalarQuantity::TetScalarQuantity(std::string name, TetMesh& mesh_,
                                     std::string definedOn_, DataType dataType_)
    : TetMeshQuantity(name, mesh_, true), dataType(dataType_),
      definedOn(definedOn_) {

    // Set the default colormap based on what kind of data is given
    cMap = defaultColorMap(dataType);
}

void TetScalarQuantity::draw() {
    if (!enabled) return;

    if (program == nullptr) {
        createProgram();
    }

    // Set uniforms
    parent.setTransformUniforms(*program);
    setProgramUniforms(*program);

    program->draw();
}
void TetScalarQuantity::geometryChanged() { program.reset(); }

void TetScalarQuantity::writeToFile(std::string filename) {
    polyscope::warning("Writing to file not yet implemented for this datatype");
}


// Update range uniforms
void TetScalarQuantity::setProgramUniforms(gl::GLProgram& program) {
    program.setUniform("u_rangeLow", vizRangeLow);
    program.setUniform("u_rangeHigh", vizRangeHigh);
}

void TetScalarQuantity::resetVizRange() {
    switch (dataType) {
    case DataType::STANDARD:
        vizRangeLow  = dataRangeLow;
        vizRangeHigh = dataRangeHigh;
        break;
    case DataType::SYMMETRIC: {
        float absRange =
            std::max(std::abs(dataRangeLow), std::abs(dataRangeHigh));
        vizRangeLow  = -absRange;
        vizRangeHigh = absRange;
    } break;
    case DataType::MAGNITUDE:
        vizRangeLow  = 0.0;
        vizRangeHigh = dataRangeHigh;
        break;
    }
}

void TetScalarQuantity::buildCustomUI() {
    ImGui::SameLine();

    // == Options popup
    if (ImGui::Button("Options")) {
        ImGui::OpenPopup("OptionsPopup");
    }
    if (ImGui::BeginPopup("OptionsPopup")) {

        if (ImGui::MenuItem("Write to file")) writeToFile();
        if (ImGui::MenuItem("Reset colormap range")) resetVizRange();

        ImGui::EndPopup();
    }

    if (buildColormapSelector(cMap)) {
        program.reset();
        hist.updateColormap(cMap);
    }

    // Draw the histogram of values
    hist.colormapRangeMin = vizRangeLow;
    hist.colormapRangeMax = vizRangeHigh;
    hist.buildUI();

    // Data range
    // Note: %g specifies are generally nicer than %e, but here we don't
    // acutally have a choice. ImGui (for somewhat valid reasons) links the
    // resolution of the slider to the decimal width of the formatted number.
    // When %g formats a number with few decimal places, sliders can break.
    // There is no way to set a minimum number of decimal places with %g,
    // unfortunately.
    {
        switch (dataType) {
        case DataType::STANDARD:
            ImGui::DragFloatRange2("", &vizRangeLow, &vizRangeHigh,
                                   (dataRangeHigh - dataRangeLow) / 100.,
                                   dataRangeLow, dataRangeHigh, "Min: %.3e",
                                   "Max: %.3e");
            break;
        case DataType::SYMMETRIC: {
            float absRange =
                std::max(std::abs(dataRangeLow), std::abs(dataRangeHigh));
            ImGui::DragFloatRange2("##range_symmetric", &vizRangeLow,
                                   &vizRangeHigh, absRange / 100., -absRange,
                                   absRange, "Min: %.3e", "Max: %.3e");
        } break;
        case DataType::MAGNITUDE: {
            ImGui::DragFloatRange2("##range_mag", &vizRangeLow, &vizRangeHigh,
                                   vizRangeHigh / 100., 0.0, dataRangeHigh,
                                   "Min: %.3e", "Max: %.3e");
        } break;
        }
    }
}

std::string TetScalarQuantity::niceName() {
    return name + " (" + definedOn + " scalar)";
}

// ========================================================
// ==========           Vertex Scalar            ==========
// ========================================================

TetVertexScalarQuantity::TetVertexScalarQuantity(std::string name,
                                                 std::vector<double> values_,
                                                 TetMesh& mesh_,
                                                 DataType dataType_)
    : TetScalarQuantity(name, mesh_, "vertex", dataType_),
      values(std::move(values_))

{
    hist.updateColormap(cMap);
    hist.buildHistogram(values, parent.vertexVolumes);

    std::tie(dataRangeLow, dataRangeHigh) = robustMinMax(values, 1e-5);
    resetVizRange();
}

void TetVertexScalarQuantity::createProgram() {
    // Create the program to draw this quantity
    program.reset(new gl::GLProgram(&gl::VERTCOLOR_SURFACE_VERT_SHADER,
                                    &gl::VERTCOLOR_SURFACE_FRAG_SHADER,
                                    gl::DrawMode::Triangles));

    // Fill color buffers
    parent.fillGeometryBuffers(*program);
    fillColorBuffers(*program);

    setMaterialForProgram(*program, "wax");
}

void TetVertexScalarQuantity::draw() {
    if (!enabled) return;

    if (program == nullptr) {
        createProgram();
    }

    // Set uniforms
    parent.setTransformUniforms(*program);
    setProgramUniforms(*program);

    program->draw();
}

void TetVertexScalarQuantity::fillColorBuffers(gl::GLProgram& p) {
    std::vector<double> colorval;
    colorval.reserve(3 * parent.nFaces());

    for (size_t iT = 0; iT < parent.tets.size(); iT++) {
        if (glm::dot(parent.tetCenters[iT], parent.sliceNormal) >
            parent.sliceDist)
            continue;
        for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            auto& face = parent.faces[iF];
            size_t D   = face.size();

            size_t vA = face[0];
            size_t vB = face[1];
            size_t vC = face[2];
            colorval.push_back(values[vA]);
            colorval.push_back(values[vB]);
            colorval.push_back(values[vC]);
        }
    }

    // Store data in buffers
    p.setAttribute("a_colorval", colorval);
    p.setTextureFromColormap("t_colormap", gl::getColorMap(cMap));
}

void TetVertexScalarQuantity::writeToFile(std::string filename) {

    throw std::runtime_error("not implemented");
}


void TetVertexScalarQuantity::buildVertexInfoGUI(size_t vInd) {
    ImGui::TextUnformatted(name.c_str());
    ImGui::NextColumn();
    ImGui::Text("%g", values[vInd]);
    ImGui::NextColumn();
}

} // namespace polyscope
