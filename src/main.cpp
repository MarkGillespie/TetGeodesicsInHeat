#include "polyscope/polyscope.h"
#include "polyscope/tet_mesh.h"

#include "tet.h"

#include "args/args.hxx"
#include "imgui.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

using namespace CompArch;

// == Geometry data
TetMesh* mesh;

// Polyscope visualization handle, to quickly add data to the surface
bool vis = true;
polyscope::TetMesh* psMesh;

float diffusionTime = 0.001;

void computeDistances(float diffusionTime) {
    std::vector<double> startingPoints(mesh->vertices.size(), 0.0);
    startingPoints[1850] = 1;
    double h             = mesh->meanEdgeLength();
    if (diffusionTime < 0) diffusionTime = h;
    std::vector<double> distances =
        mesh->distances(startingPoints, diffusionTime);
    if (vis) {
        auto* q = psMesh->addVertexScalarQuantity("distances", distances);
        q->setColorMap(polyscope::gl::ColorMapID::STRIPES);
        q->setEnabled(true);
    }
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {
    if (ImGui::SliderFloat("diffusion time", &diffusionTime, 0.0f, 1.0f,
                           "%.3f")) {
        computeDistances(diffusionTime);
    }
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Geometry program");
    args::Positional<std::string> inputFilename(
        parser, "mesh", "Tet mesh (ele file) to be processed.");
    args::Flag noVis(parser, "noVis", "Set to disable visualization",
                     {'n', "no_vis"});

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
    if (noVis) {
        vis = false;
    }

    mesh = TetMesh::loadFromFile(filename);

    std::vector<glm::vec3> faceNormals;
    for (auto f : mesh->faceList()) {
        Vector3 a = mesh->vertices[f[0]].position;
        Vector3 b = mesh->vertices[f[1]].position;
        Vector3 c = mesh->vertices[f[2]].position;

        Vector3 N = cross(b - a, c - a);
        N /= N.norm();
        faceNormals.emplace_back(glm::vec3{N.x, N.y, N.z});
    }

    if (vis) {
        // Initialize polyscope
        polyscope::init();

        // Set the callback function
        polyscope::state::userCallback = myCallback;

        // Register the mesh with polyscope
        psMesh = polyscope::registerTetMesh("tMesh", mesh->vertexPositions(),
                                            mesh->tetList());
        polyscope::getTetMesh("tMesh");
        psMesh->addFaceVectorQuantity("normal", faceNormals);
        psMesh->addVertexScalarQuantity("volumes", mesh->vertexDualVolumes);
    }

    computeDistances(-1);

    if (vis) {
        // Give control to the polyscope gui
        polyscope::show();
    }

    return EXIT_SUCCESS;
}
