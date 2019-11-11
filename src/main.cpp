#include "tet.h"

#include "args.hxx"

#include <ctime>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

using namespace CompArch;

// == Geometry data
TetMesh* mesh;

float diffusionTime = 0.001;

void computeDistances(float diffusionTime) {
    std::vector<double> startingPoints(mesh->vertices.size(), 0.0);
    startingPoints[8528] = 1;
    double h             = mesh->meanEdgeLength();
    if (diffusionTime < 0) diffusionTime = h;
    std::vector<double> distances =
        mesh->distances(startingPoints, diffusionTime);
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Geometry program");
    args::Positional<std::string> inputFilename(
        parser, "mesh", "Tet mesh (ele file) to be processed.");

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

    Eigen::SparseMatrix<double> L    = mesh->weakLaplacian();
    Eigen::SparseMatrix<double> M    = mesh->massMatrix();
    Eigen::SparseMatrix<double> flow = M + 0.1 * L;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    std::clock_t start;
    double duration;

    start = std::clock();

    solver.compute(flow);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000;

    std::cout<< mesh->tets.size()<<"\t"<< duration <<"\n";

    return EXIT_SUCCESS;
}
