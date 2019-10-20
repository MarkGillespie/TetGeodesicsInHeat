#pragma once

#include "polyscope/color_management.h"
#include "polyscope/gl/materials/materials.h"
#include "polyscope/gl/shaders/surface_shaders.h"
#include "polyscope/gl/shaders/wireframe_shaders.h"
#include "polyscope/pick.h"
#include "polyscope/polyscope.h"
#include "polyscope/structure.h"
#include "polyscope/surface_mesh.h"

#include "tet_scalar_quantity.h"
#include "tet_vector_quantity.h"

#include "glm/glm.hpp"

#include <iostream>
#include <string>

namespace polyscope {

// Forward declarations for quantities
class TetVertexScalarQuantity;
class TetFaceScalarQuantity;
class TetVertexVectorQuantity;
class TetFaceVectorQuantity;

template <> // Specialize the quantity type
struct QuantityTypeHelper<TetMesh> {
    typedef TetMeshQuantity type;
};


class TetMesh : public QuantityStructure<TetMesh> {
  public:
    typedef TetMeshQuantity QuantityType;

    TetMesh(std::string name_);
    TetMesh(std::string name_, std::vector<glm::vec3> vertices_,
            std::vector<std::vector<size_t>> tets_);

    std::vector<glm::vec3> vertices;
    std::vector<std::vector<size_t>> tets;

    std::vector<std::vector<size_t>> faces;
    std::vector<glm::vec3> faceNormals;
    std::vector<glm::vec3> tetCenters;
    glm::vec3 faceCenter(size_t iF);

    std::vector<double> vertexVolumes;
    std::vector<double> faceAreas;
    size_t nVertices();
    size_t nFaces();

    void computeGeometryData();

    // == Render the the structure on screen
    void draw();
    void drawPick();

    // == Build the ImGUI ui elements
    void buildCustomUI(); // overridden by childen to add custom UI data
    void
    buildCustomOptionsUI(); // overridden by childen to add to the options menu
    void buildPickUI(size_t localPickID); // Draw pick UI elements when index
                                          // localPickID is selected

    // = Identifying data
    const std::string
        name; // should be unique amongst registered structures with this type

    // = Length and bounding box (returned in object coordinates)
    std::tuple<glm::vec3, glm::vec3>
    boundingBox();        // get axis-aligned bounding box
    double lengthScale(); // get characteristic length

    std::string typeName();

    // = Scene transform
    glm::mat4 objectTransform = glm::mat4(1.0);
    void centerBoundingBox();

    // = Drawing related things
    std::unique_ptr<gl::GLProgram> program;
    std::unique_ptr<gl::GLProgram> pickProgram;
    std::unique_ptr<gl::GLProgram> wireframeProgram;
    void prepare();
    void prepareWireframe();
    void preparePick();
    void geometryChanged(); // call whenever geometry changed
    void refillBuffers();
    void fillGeometryBuffers(gl::GLProgram& p);
    void fillGeometryBuffersWireframe(gl::GLProgram& p);
    void fillGeometryBuffersPick(gl::GLProgram& p);

    // = Visualization settings
    glm::vec3 baseColor;
    glm::vec3 tetColor;
    glm::vec3 edgeColor;
    float edgeWidth = 1.0;
    glm::vec3 sliceNormal{1, 0, 0};
    float sliceDist  = 1.0;
    float sliceTheta = 0.0;
    float slicePhi   = 0.0;

    // Picking-related
    // Order of indexing: vertices, faces
    // TODO: add edges
    // Within each set, uses the implicit ordering from the mesh data structure
    // These starts are LOCAL indices, indexing elements only with the mesh
    size_t facePickIndStart;
    void buildVertexInfoGui(size_t vInd);
    void buildFaceInfoGui(size_t fInd);

    template <class T>
    TetVertexScalarQuantity*
    addVertexScalarQuantity(std::string name, const T& data,
                            DataType type = DataType::STANDARD) {
        validateSize(data, vertices.size(), "vertex scalar quantity " + name);
        return addVertexScalarQuantityImpl(
            name, standardizeArray<double, T>(data), type);
    }
    template <class T>
    TetFaceScalarQuantity*
    addFaceScalarQuantity(std::string name, const T& data,
                          DataType type = DataType::STANDARD) {
        validateSize(data, faces.size(), "face scalar quantity " + name);
        return addFaceScalarQuantityImpl(
            name, standardizeArray<double, T>(data), type);
    }
    template <class T>
    TetVertexVectorQuantity*
    addVertexVectorQuantity(std::string name, const T& vectors,
                            VectorType vectorType = VectorType::STANDARD) {
        validateSize(vectors, vertices.size(),
                     "vertex vector quantity " + name);
        return addVertexVectorQuantityImpl(
            name, standardizeVectorArray<glm::vec3, 3>(vectors), vectorType);
    }
    template <class T>
    TetFaceVectorQuantity*
    addFaceVectorQuantity(std::string name, const T& vectors,
                          VectorType vectorType = VectorType::STANDARD) {
        validateSize(vectors, faces.size(), "face vector quantity " + name);
        return addFaceVectorQuantityImpl(
            name, standardizeVectorArray<glm::vec3, 3>(vectors), vectorType);
    }


    TetVertexScalarQuantity*
    addVertexScalarQuantityImpl(std::string name,
                                const std::vector<double>& data, DataType type);
    TetFaceScalarQuantity*
    addFaceScalarQuantityImpl(std::string name, const std::vector<double>& data,
                              DataType type);
    TetVertexVectorQuantity*
    addVertexVectorQuantityImpl(std::string name,
                                const std::vector<glm::vec3>& vectors,
                                VectorType vectorType);
    TetFaceVectorQuantity*
    addFaceVectorQuantityImpl(std::string name,
                              const std::vector<glm::vec3>& vectors,
                              VectorType vectorType);
    // === Member variables ===
    static const std::string structureTypeName;
};

// Register functions
template <class V, class T>
TetMesh* registerTetMesh(std::string name, const V& vertexPositions,
                         const T& tetIndices, bool replaceIfPresent = true) {
    TetMesh* t =
        new TetMesh(name, standardizeVectorArray<glm::vec3, 3>(vertexPositions),
                    standardizeNestedList<size_t, T>(tetIndices));
    bool success = registerStructure(t);
    if (!success) {
        safeDelete(t);
    }

    return t;
}

// Shorthand to get a mesh from polyscope
inline TetMesh* getTetMesh(std::string name = "") {
    return dynamic_cast<TetMesh*>(
        getStructure(TetMesh::structureTypeName, name));
}

} // namespace polyscope
