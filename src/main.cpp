#include "polyscope/polyscope.h"
#include "polyscope/tet_mesh.h"

#include "cluster.h"
#include "tet.h"

#include "args/args.hxx"
#include "imgui.h"

#include <ctime>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

using namespace CompArch;

// == Geometry data
TetMesh* mesh;

// Polyscope visualization handle, to quickly add data to the surface
bool vis = true;
polyscope::TetMesh* psMesh;

float diffusionTime = 0.001;

void computeDistances(float diffusionTime, bool verbose = false) {
    if (verbose) cout << "Beginning to compute distances" << endl;
    std::vector<double> startingPoints(mesh->vertices.size(), 0.0);
    startingPoints[1850] = 1;
    if (verbose) cout << "about to compute h" << endl;
    double h = mesh->meanEdgeLength();
    if (diffusionTime < 0) diffusionTime = h;
    if (verbose) cout << "done computing h" << endl;
    std::vector<double> distances =
        mesh->distances(startingPoints, diffusionTime, verbose);

    if (vis) {
        auto* q = psMesh->addVertexScalarQuantity("distances", distances);
    }
    // q->setEnabled(true);
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
    cout << "Loaded mesh" << endl;

    if (vis) {
        // Initialize polyscope
        polyscope::init();

        // Set the callback function
        polyscope::state::userCallback = myCallback;

        // Register the mesh with polyscope
        psMesh =
            polyscope::registerTetMesh("tMesh", mesh->vertexPositions(),
                                       mesh->tetList(), mesh->neighborList());
        polyscope::getTetMesh("tMesh");
        computeDistances(-1);

        std::vector<std::vector<size_t>> clusters = cluster(*mesh, 256);
        size_t nC                                 = clusters.size();
        size_t period                             = 37;
        std::vector<int> clusterIndex(mesh->vertices.size(), -1);
        double avgSize = 0;
        for (size_t iC = 0; iC < clusters.size(); ++iC) {
            std::vector<size_t> cluster = clusters[iC];
            avgSize += cluster.size();
            for (size_t iV : cluster) {
                clusterIndex[iV] = (37 * iC) % nC;
            }
        }
        avgSize /= nC;

        cout << "nClusters: " << nC << "\t avg cluster size: " << avgSize
             << endl;


        psMesh->addVertexScalarQuantity("cluster", clusterIndex);
    }

    if (vis) {
        // Give control to the polyscope gui
        polyscope::show();
    }

    return EXIT_SUCCESS;
}
