#pragma once

#include "polyscope/quantity.h"
#include "polyscope/structure.h"


namespace polyscope {

// Forward delcare surface mesh
class TetMesh;

// Extend Quantity<TetMesh> to add a few extra functions
class TetMeshQuantity : public Quantity<TetMesh> {
  public:
    TetMeshQuantity(std::string name, TetMesh& parentStructure,
                    bool dominates = false);
    ~TetMeshQuantity(){};

  public:
    // Notify that the geometry has changed
    virtual void geometryChanged();

    // Build GUI info about this element
    virtual void buildVertexInfoGUI(size_t vInd);
    virtual void buildFaceInfoGUI(size_t fInd);
    virtual void buildEdgeInfoGUI(size_t eInd);
    virtual void buildHalfedgeInfoGUI(size_t heInd);
    virtual void buildTetInfoGUI(size_t tInd);
};

} // namespace polyscope
