#include "tet.h"

#include "args.hxx"

#include "cuda/cg.cuh"

#include <ctime>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

using namespace CompArch;

// == Geometry data
TetMesh* mesh;

float diffusionTime = 0.001;

void testSolver(size_t startIndex, double t, bool useCSR = false) {
    std::vector<double> distances;
    distances.reserve(mesh->vertices.size());

    std::vector<double> start(mesh->vertices.size(), 0.0);
    start[startIndex] = 1;
    if (t < 0) t = mesh->meanEdgeLength();

    Eigen::VectorXd u0 = Eigen::VectorXd::Map(start.data(), start.size());

    Eigen::VectorXd u(mesh->vertices.size());
    Eigen::VectorXd phi(mesh->vertices.size());
    Eigen::VectorXd divX = Eigen::VectorXd::Random(mesh->vertices.size());
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(divX.size());
    ones.normalize();
    divX -= divX.dot(ones) * ones;


    if (useCSR) {
        cgSolveCSR(u, u0, *mesh, 1e-8, t);
        cgSolveCSR(phi, divX, *mesh, 1e-8, -1);
    } else {
        cgSolve(u, u0, *mesh, 1e-8, t);
        cgSolve(phi, divX, *mesh, 1e-8, -1);
    }

    Eigen::SparseMatrix<double> L    = mesh->weakLaplacian();
    Eigen::SparseMatrix<double> M    = mesh->massMatrix();

    Eigen::SparseMatrix<double> flow = M + t * L;
    cout << "Residual: " << (flow * u  - u0).norm();
    cout << "\tResidual 2: " << (L * phi - divX).norm() << endl;
}

std::vector<double> computeDistances(size_t startIndex, double t, bool useCUDA, bool useCSR=false) {
    std::vector<double> distances;
    distances.reserve(mesh->vertices.size());

    std::vector<double> start(mesh->vertices.size(), 0.0);
    start[startIndex] = 1;
    if (t < 0) t = mesh->meanEdgeLength();

    Eigen::VectorXd u0 = Eigen::VectorXd::Map(start.data(), start.size());
    Eigen::SparseMatrix<double> L    = mesh->weakLaplacian();
    Eigen::SparseMatrix<double> M    = mesh->massMatrix();

    Eigen::VectorXd u(mesh->vertices.size());
    Eigen::SparseMatrix<double> flow = M + t * L;
    if (useCUDA) {
        if (useCSR) {
            cgSolveCSR(u, u0, *mesh, 1e-8, t);
        } else {
            cgSolve(u, u0, *mesh, 1e-8, t);
        }
        double residual = (flow * u - u0).norm();
        if (residual > 1e-5)
            cout << "Residual 1: " << residual << endl;
    } else {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(flow);
        u = solver.solve(u0);
    }

    Eigen::VectorXd divX = Eigen::VectorXd::Zero(u.size());

    std::vector<Vector3> tetXs;
    for (Tet t : mesh->tets) {
        std::array<Vector3, 4> vertexPositions = mesh->layOutIntrinsicTet(t);

        std::array<double, 4> tetU{u[t.verts[0]], u[t.verts[1]], u[t.verts[2]],
                                   u[t.verts[3]]};
        Vector3 tetGradU = grad(tetU, vertexPositions);
        Vector3 X = tetGradU.normalize();

        tetXs.emplace_back(Vector3{X.x, X.y, X.z});

        std::array<double, 4> tetDivX = div(X, vertexPositions);
        for (size_t i = 0; i < 4; ++i) {
            divX[t.verts[i]] += tetDivX[i];
        }
    }

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(divX.size());
    ones.normalize();
    divX -= divX.dot(ones) * ones;

    Eigen::VectorXd phi(mesh->vertices.size());
    if (useCUDA) {
        if (useCSR) {
            cgSolveCSR(phi, divX, *mesh, 1e-8, -1);
        } else {
            cgSolve(phi, divX, *mesh, 1e-8, -1);
        }
        double residual = (L * phi - divX).norm();
        if (residual > 1e-5)
            cout << "Residual 2: " << residual << endl;
    } else {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(L);
        phi = solver.solve(divX);
    }

    for (int i = 0; i < phi.size(); ++i) {
        distances[i] = phi[i];
    }

    double minDist = distances[0];
    for (size_t i = 1; i < distances.size(); ++i) {
        minDist = fmin(minDist, distances[i]);
    }
    for (size_t i = 0; i < distances.size(); ++i) {
        distances[i] -= minDist;
        assert(distances[i] >= 0);
    }

    return distances;
}

int main(int argc, char** argv) {

    // Configure the argument parser
    args::ArgumentParser parser("Geometry program");
    args::Positional<std::string> inputFilename(
        parser, "mesh", "Tet mesh (ele file) to be processed.");
    args::Positional<std::string> niceName(
        parser, "name", "Nice name for printed output.");

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

    std::string descriptionName = filename;
    if (niceName) {
        descriptionName = args::get(niceName);
    }

    mesh = TetMesh::loadFromFile(filename);
    std::cout << descriptionName << "\t" << mesh->tets.size();

    std::cout << endl;
    std::cout << "CSR test: " ;
    testSolver(0, -1, true);
    std::cout << "non CSR test: ";
    testSolver(0, -1, false);
    std::cout << "Done testing " << endl;

    std::clock_t start;
    double duration;

    start = std::clock();
    computeDistances(0, -1, false);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000;
    std::cout<< "\t" << duration;

    start = std::clock();
    computeDistances(0, -1, true, true);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000;
    std::cout<< "\tCSR: " << duration;

    start = std::clock();
    computeDistances(0, -1, true, false);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000;
    std::cout<< "\tmine: " << duration;

    std::cout << std::endl;

    return EXIT_SUCCESS;
}
