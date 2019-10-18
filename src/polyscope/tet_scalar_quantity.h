#pragma once

#include "polyscope/affine_remapper.h"
#include "polyscope/gl/color_maps.h"
#include "polyscope/histogram.h"
#include "polyscope/quantity.h"
#include "polyscope/structure.h"

#include "tet_mesh.h"

namespace polyscope {

// Forward declare TetMesh
class TetMesh;

class TetScalarQuantity : public Quantity<TetMesh> {
public:
  TetScalarQuantity(std::string name, TetMesh& mesh_, std::string definedOn, DataType dataType);

  virtual void draw() override;
  virtual void buildCustomUI() override;
  virtual std::string niceName() override;

  virtual void writeToFile(std::string filename = "");

  // === Members
  const DataType dataType;

protected:
  // Affine data maps and limits
  void resetVizRange();
  float vizRangeLow, vizRangeHigh;
  float dataRangeHigh, dataRangeLow;
  Histogram hist;

  // UI internals
  gl::ColorMapID cMap;
  const std::string definedOn;
  std::unique_ptr<gl::GLProgram> program;

  // Helpers
  virtual void createProgram() = 0;
  void setProgramUniforms(gl::GLProgram& program);
};

// ========================================================
// ==========           Vertex Scalar            ==========
// ========================================================

class TetVertexScalarQuantity : public TetScalarQuantity {
public:
  TetVertexScalarQuantity(std::string name, std::vector<double> values_, TetMesh& mesh_,
                              DataType dataType_ = DataType::STANDARD);

  virtual void createProgram() override;

  void fillColorBuffers(gl::GLProgram& p);

  void draw() override;

  //void buildVertexInfoGUI(size_t vInd) override;
  virtual void writeToFile(std::string filename = "") override;

  // === Members
  std::vector<double> values;
};
}//polyscope
