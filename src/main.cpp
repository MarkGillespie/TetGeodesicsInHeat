#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/tet_mesh.h"

#include "tet.h"

#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
TetMesh* mesh;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::TetMesh *psMesh;

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {}

int main(int argc, char **argv) {

  // Configure the argument parser
  args::ArgumentParser parser("Geometry program");
  args::Positional<std::string> inputFilename(parser, "mesh",
                                              "Tet mesh (ele file) to be processed.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  std::string filename = "../../meshes/TetMeshes/bunny_small.1.ele";
  // Make sure a mesh name was given
  if (inputFilename) {
    filename = args::get(inputFilename);
  }


  mesh = TetMesh::loadFromFile(filename);
  std::vector<glm::vec3> pos = mesh->vertexPositions();

  std::vector<double> xData, yData, zData;
  for (glm::vec3 v : pos) {
      xData.emplace_back(v.x);
      yData.emplace_back(v.y);
      zData.emplace_back(v.z);
  }

  std::vector<glm::vec3> faceNormals;
  for (auto f : mesh->faceList()) {
      Vector3 a = mesh->vertices[f[0]].position;
      Vector3 b = mesh->vertices[f[1]].position;
      Vector3 c = mesh->vertices[f[2]].position;

      Vector3 N = cross(b-a, c-a);
      N /= N.norm();
      faceNormals.emplace_back(glm::vec3{N.x, N.y, N.z});
  }


  // Initialize polyscope
  polyscope::init();

  // Set the callback function
  polyscope::state::userCallback = myCallback;

  // Register the mesh with polyscope
  psMesh = polyscope::registerTetMesh(
      polyscope::guessNiceNameFromPath(filename),
      mesh->vertexPositions(), mesh->tetList());
  psMesh->addVertexScalarQuantity("x", xData);
  psMesh->addVertexScalarQuantity("y", yData);
  psMesh->addVertexScalarQuantity("z", zData);
  psMesh->addFaceVectorQuantity("normal", faceNormals);


  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
