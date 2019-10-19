#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/tet_mesh.h"

#include "tet.h"

#include "args/args.hxx"
#include "imgui.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace CompArch;

std::vector<Eigen::VectorXd> gEig(Eigen::SparseMatrix<double> A, Eigen::SparseMatrix<double> B, size_t n = 1) {

  std::vector<Eigen::VectorXd> evecs;
  evecs.emplace_back(Eigen::VectorXd::Ones(A.cols()));
  evecs[0] /= sqrt(evecs[0].dot(B * evecs[0]));
  // evecs[0] /= sqrt(evecs[0].dot(evecs[0]));

  for (size_t iter = 1; iter < n; ++iter) {
    Eigen::VectorXd v = Eigen::VectorXd::Random(A.cols());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(A);

    for (size_t i = 0; i < 100; ++i) {
      // v = solver.solve(v);
      v = solver.solve(B * v);
      for (size_t prev = 0; prev < iter; ++prev) {
        assert(v.rows() == evecs[prev].rows());
        // v = v - v.dot(evecs[prev]) * evecs[prev];
        v = v - v.dot(B * evecs[prev]) * evecs[prev];
      }
      v /= sqrt(v.dot(B * v));
      // v /= sqrt(v.dot(v));
    }

    for (size_t prev = 0; prev < iter; ++prev) {
      assert(abs(v.dot(B * evecs[prev])) < 1e-4);
      // assert(abs(v.dot(evecs[prev])) < 1e-4);
    }
    evecs.emplace_back(v);

  }

  for (size_t i = 0; i < evecs.size(); ++i) {
    Eigen::VectorXd v = evecs[i];
    double lambda = v.dot(A*v) / v.dot(B * v);
    double lambda1 = (A*v)[0] / (B*v)[0];
    double lambda2 = (A*v)[1] / (B*v)[1];
    assert(abs(lambda1-lambda2) < 1e-4);
    // TODO: figure out what to assert here
    assert(abs((lambda * B * v - A*v).norm()) < 1e-4);
    // double lambda = v.dot(A*v) / v.dot(v);
    // assert(abs((lambda * v - A*v).norm()) < 1e-4);
    cout << "λ: " << lambda << endl;
  }

  return evecs;
}

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

  Eigen::SparseMatrix<double> L = mesh->weakLaplacian();
  Eigen::SparseMatrix<double> M = mesh->massMatrix();

  std::vector<Eigen::VectorXd> evecs = gEig(L, M, 5);

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

  for (size_t i = 0; i < evecs.size(); ++i) {
    psMesh->addVertexScalarQuantity("evec " + std::to_string(i), evecs[i]);
  }


  /*
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(-2, 2);


  std::vector<Vector3> positions;
  positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});
  positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});
  positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});
  positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});

  for (size_t i = 0; i < 4; ++i) {
    positions.emplace_back(positions[i] + Vector3{4, 0, 0});
  }

  std::vector<std::vector<size_t>> tets;
  tets.emplace_back(std::vector<size_t>{0, 1, 2, 3});
  tets.emplace_back(std::vector<size_t>{5, 4, 6, 7});
  std::vector<std::vector<size_t>> neigh;
  neigh.emplace_back(std::vector<size_t>{0, 0, 0, 0});
  neigh.emplace_back(std::vector<size_t>{0, 0, 0, 0});

  TetMesh* singleTetMesh = TetMesh::construct(positions, tets, neigh);

  polyscope::TetMesh* psSingleTetMesh = polyscope::registerTetMesh(
                                                                   "Single Tet",
      singleTetMesh->vertexPositions(), singleTetMesh->tetList());

  std::vector<glm::vec3> moreFaceNormals;
  for (auto f : singleTetMesh->faceList()) {
    Vector3 a = singleTetMesh->vertices[f[0]].position;
    Vector3 b = singleTetMesh->vertices[f[1]].position;
    Vector3 c = singleTetMesh->vertices[f[2]].position;

    Vector3 N = cross(b-a, c-a);
    N /= N.norm();
    moreFaceNormals.emplace_back(glm::vec3{N.x, N.y, N.z});
  }
  psSingleTetMesh->addFaceVectorQuantity("normal", moreFaceNormals);
  */

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
