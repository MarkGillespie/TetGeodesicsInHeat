#pragma once

#include "polyscope/affine_remapper.h"
#include "polyscope/ribbon_artist.h"

#include "tet_mesh.h"
#include "tet_mesh_quantity.h"

namespace polyscope {

// Forward declare TetMesh
class TetMesh;

// ==== Common base class

// Represents a general vector field associated with a tet mesh, including
// R3 fields in the ambient space and R2 fields embedded in the tet
class TetVectorQuantity : public TetMeshQuantity {
  public:
    TetVectorQuantity(std::string name, TetMesh& mesh_, MeshElement definedOn_,
                      VectorType vectorType_ = VectorType::STANDARD);


    virtual void draw() override;
    virtual void buildCustomUI() override;
    virtual void geometryChanged() override;
    virtual bool skipVector(size_t iV) = 0;

    // Allow children to append to the UI
    virtual void drawSubUI();

    // === Members
    const VectorType vectorType;
    std::vector<glm::vec3> vectorRoots;
    std::vector<glm::vec3> vectors;
    float lengthMult; // longest vector will be this fraction of lengthScale (if
                      // not ambient)
    float radiusMult; // radius is this fraction of lengthScale
    glm::vec3 vectorColor;
    MeshElement definedOn;

    // A ribbon viz that is appropriate for some fields
    std::unique_ptr<RibbonArtist> ribbonArtist;
    bool ribbonEnabled = false;

    // Clip vector field when clipping tets
    bool hideWithMesh = true;

    // The map that takes values to [0,1] for drawing
    AffineRemapper<glm::vec3> mapper;

    void writeToFile(std::string filename = "");

    // GL things
    void prepareProgram();
    std::unique_ptr<gl::GLProgram> program;

  protected:
    // Set up the mapper for vectors
    void prepareVectorMapper();
};


// ==== R3 vectors at vertices

class TetVertexVectorQuantity : public TetVectorQuantity {
  public:
    TetVertexVectorQuantity(std::string name, std::vector<glm::vec3> vectors_,
                            TetMesh& mesh_,
                            VectorType vectorType_ = VectorType::STANDARD);

    std::vector<glm::vec3> vectorField;
    virtual std::string niceName() override;
    virtual void buildVertexInfoGUI(size_t vInd) override;
    virtual bool skipVector(size_t iV) override;
};

// ==== R3 vectors at faces

class TetFaceVectorQuantity : public TetVectorQuantity {
  public:
    TetFaceVectorQuantity(std::string name, std::vector<glm::vec3> vectors_,
                          TetMesh& mesh_,
                          VectorType vectorType_ = VectorType::STANDARD);

    std::vector<glm::vec3> vectorField;

    virtual std::string niceName() override;
    virtual void buildFaceInfoGUI(size_t fInd) override;
    virtual bool skipVector(size_t iV) override;
};

// ==== R3 vectors at tets

class TetTetVectorQuantity : public TetVectorQuantity {
  public:
    TetTetVectorQuantity(std::string name, std::vector<glm::vec3> vectors_,
                         TetMesh& mesh_,
                         VectorType vectorType_ = VectorType::STANDARD);

    std::vector<glm::vec3> vectorField;

    virtual std::string niceName() override;
    virtual void buildTetInfoGUI(size_t tInd) override;
    virtual bool skipVector(size_t iV) override;
};

} // namespace polyscope
