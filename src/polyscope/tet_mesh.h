#pragma once

#include "polyscope/polyscope.h"
#include "polyscope/structure.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/color_management.h"
#include "polyscope/gl/materials/materials.h"
#include "polyscope/gl/shaders/surface_shaders.h"
#include "polyscope/gl/shaders/wireframe_shaders.h"

#include "tet_scalar_quantity.h"

#include "glm/glm.hpp"

#include <iostream>
#include <string>

namespace polyscope{

// Forward declarations for quantities
class TetVertexScalarQuantity;
class TetFaceVectorQuantity;

class TetMesh : public QuantityStructure<TetMesh> {
    public:
        typedef Quantity<TetMesh> QuantityType;

        TetMesh(std::string name_);
        TetMesh(std::string name_, std::vector<glm::vec3> vertices_, std::vector<std::vector<size_t>> tets_);

        std::vector<glm::vec3> vertices;
        std::vector<std::vector<size_t>> tets;

        std::vector<std::vector<size_t>> faces;
        std::vector<glm::vec3> faceNormals;
        std::vector<glm::vec3> tetCenters;

        std::vector<double> vertexVolumes;
        size_t nFaces();

        void computeGeometryData();

        // == Render the the structure on screen
        void draw();
        void drawPick();
      
        // == Build the ImGUI ui elements
        void buildUI();
        void buildCustomUI();      // overridden by childen to add custom UI data
        void buildCustomOptionsUI();   // overridden by childen to add to the options menu
        void buildPickUI(size_t localPickID); // Draw pick UI elements when index localPickID is selected
      
        // = Identifying data
        const std::string name; // should be unique amongst registered structures with this type
      
        // = Length and bounding box (returned in object coordinates)
        std::tuple<glm::vec3, glm::vec3> boundingBox(); // get axis-aligned bounding box
        double lengthScale();                           // get characteristic length
      
        std::string typeName();
      
        // = Scene transform
        glm::mat4 objectTransform = glm::mat4(1.0);
        void centerBoundingBox();

        // = Drawing related things
        bool quantitiesMustRefillBuffers = false;
        std::unique_ptr<gl::GLProgram> program;
        std::unique_ptr<gl::GLProgram> wireframeProgram;
        void prepare();
        void prepareWireframe();
        void refillBuffers();
        void fillGeometryBuffers(gl::GLProgram& p);
        void fillGeometryBuffersWireframe(gl::GLProgram& p);

        // = Visualization settings
        glm::vec3 baseColor;
        glm::vec3 tetColor;
        glm::vec3 edgeColor;
        float edgeWidth = 1.0;
        glm::vec3 sliceNormal{1, 0, 0};
        float sliceDist = 1.0;
        float sliceTheta = 0.0;
        float slicePhi = 0.0;

        template <class T>
        TetVertexScalarQuantity* addVertexScalarQuantity(std::string name, const T& data, DataType type = DataType::STANDARD) {
            validateSize(data, vertices.size(), "vertex scalar quantity " + name);
            return addVertexScalarQuantityImpl(name, standardizeArray<double, T>(data), type);
        }
        TetVertexScalarQuantity* addVertexScalarQuantityImpl(std::string name, const std::vector<double>& data, DataType type);
  
    protected:
        // = State
        bool enabled = true;
};

// Register functions
template <class V, class T>
TetMesh* registerTetMesh(std::string name, const V& vertexPositions, const T& tetIndices,
                                 bool replaceIfPresent = true) {
  TetMesh* t = new TetMesh(name, standardizeVectorArray<glm::vec3, 3>(vertexPositions),
                                   standardizeNestedList<size_t, T>(tetIndices));
  bool success = registerStructure(t);
  if (!success) {
    safeDelete(t);
  }

  return t;
}

} // polyscope
