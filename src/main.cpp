#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "tet.h"

#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
TetMesh* mesh;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh *psMesh;

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
  cout << "nVertices: " << mesh->vertices.size() << endl;
  cout << "nTets:   : " << mesh->tets.size() << endl;

  std::vector<glm::vec3> pos = mesh->vertexPositions();

  std::vector<std::vector<size_t>> faces = mesh->faces();
  cout << "nVertices: " << pos.size() << endl;
  cout << "nFaces:   : " << faces.size() << endl;

  for (size_t i = 0; i < 5; ++i) {
      assert(i < pos.size());
      cout << "Vertex: <" << pos[i].x << ", " << pos[i].y << " " << pos[i].z << ">" << endl;
  }
  cout << endl;
  for (size_t j = 0; j < 5; ++j) {
      cout << "FACE " << j << endl;
      for (size_t i : faces[j]) {
          cout << "\t" << i << ": " << std::flush;
          assert(i < pos.size());
          cout << "<" << pos[i].x << ", " << pos[i].y << " " << pos[i].z << ">, ";
      }
      cout << endl;
  }


// Initialize polyscope
  polyscope::init();

  // Set the callback function
  polyscope::state::userCallback = myCallback;

  // Register the mesh with polyscope
  psMesh = polyscope::registerSurfaceMesh(
      polyscope::guessNiceNameFromPath(filename),
      mesh->vertexPositions(), mesh->faces());

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
