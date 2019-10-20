#include "test_utils.h"
#include "tet.h"

#include <random>

class TetTest : public ::testing::Test {
  public:
    static CompArch::TetMesh* tetMesh;

  protected:
    static void SetUpTestSuite() {

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-2, 2);


        std::vector<Vector3> positions;
        /* positions.emplace_back(Vector3{0, 0, 0}); */
        /* positions.emplace_back(Vector3{1, 0, 0}); */
        /* positions.emplace_back(Vector3{0, 1, 0}); */
        /* positions.emplace_back(Vector3{0, 0, 1}); */
        positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});
        positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});
        positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});
        positions.emplace_back(Vector3{dist(e2), dist(e2), dist(e2)});

        std::vector<std::vector<size_t>> tets;
        tets.emplace_back(std::vector<size_t>{0, 1, 2, 3});

        std::vector<std::vector<size_t>> neigh;
        neigh.emplace_back(std::vector<size_t>{0, 0, 0, 0});

        tetMesh = CompArch::TetMesh::construct(positions, tets, neigh);
    }

    void SetUp() override {}
};

CompArch::TetMesh* TetTest::tetMesh = nullptr;

TEST_F(TetTest, volumeFormulaTest) {
    CompArch::Tet t = tetMesh->tets[0];
    double vol      = tetMesh->tetVolume(t);

    Vector3 v0       = tetMesh->vertices[t.verts[0]].position;
    Vector3 v1       = tetMesh->vertices[t.verts[1]].position;
    Vector3 v2       = tetMesh->vertices[t.verts[2]].position;
    Vector3 v3       = tetMesh->vertices[t.verts[3]].position;
    double simpleVol = dot(v3 - v0, cross(v1 - v0, v2 - v0)) / 6;

    EXPECT_FLOAT_EQ(vol, abs(simpleVol));
}

TEST_F(TetTest, dihedralAngleFormulaTest) {
    CompArch::Tet t = tetMesh->tets[0];

    Vector3 v0 = tetMesh->vertices[t.verts[0]].position;
    Vector3 v1 = tetMesh->vertices[t.verts[1]].position;
    Vector3 v2 = tetMesh->vertices[t.verts[2]].position;
    Vector3 v3 = tetMesh->vertices[t.verts[3]].position;

    Vector3 n012 = cross(v1 - v0, v2 - v0).normalize();
    Vector3 n023 = cross(v2 - v0, v3 - v0).normalize();
    Vector3 n031 = cross(v3 - v0, v1 - v0).normalize();
    Vector3 n213 = cross(v1 - v2, v3 - v2).normalize();

    double alpha01 = PI - acos(dot(n012, n031));
    double alpha02 = PI - acos(dot(n012, n023));
    double alpha03 = PI - acos(dot(n023, n031));
    double alpha12 = PI - acos(dot(n012, n213));
    double alpha13 = PI - acos(dot(n031, n213));
    double alpha23 = PI - acos(dot(n023, n213));

    std::vector<double> angles = tetMesh->dihedralAngles(t);
    EXPECT_FLOAT_EQ(alpha01, angles[0]);
    EXPECT_FLOAT_EQ(alpha02, angles[1]);
    EXPECT_FLOAT_EQ(alpha03, angles[2]);
    EXPECT_FLOAT_EQ(alpha12, angles[3]);
    EXPECT_FLOAT_EQ(alpha13, angles[4]);
    EXPECT_FLOAT_EQ(alpha23, angles[5]);
}

TEST_F(TetTest, layOutTet) {

    CompArch::Tet t        = tetMesh->tets[0];
    std::vector<Vector3> p = tetMesh->layOutIntrinsicTet(t);
    EXPECT_GE(dot(p[3] - p[0], cross(p[1] - p[0], p[2] - p[0])), 0);

    Vector3 p0 = tetMesh->vertices[t.verts[0]].position;
    Vector3 p1 = tetMesh->vertices[t.verts[1]].position;
    Vector3 p2 = tetMesh->vertices[t.verts[2]].position;
    Vector3 p3 = tetMesh->vertices[t.verts[3]].position;

    double u0 = tetMesh->scaleFactors[t.verts[0]];
    double u1 = tetMesh->scaleFactors[t.verts[1]];
    double u2 = tetMesh->scaleFactors[t.verts[2]];
    double u3 = tetMesh->scaleFactors[t.verts[3]];

    double e01 = norm(p0 - p1) * exp(0.5 * (u0 + u1));
    double e02 = norm(p0 - p2) * exp(0.5 * (u0 + u2));
    double e03 = norm(p0 - p3) * exp(0.5 * (u0 + u3));
    double e12 = norm(p1 - p2) * exp(0.5 * (u1 + u2));
    double e13 = norm(p1 - p3) * exp(0.5 * (u1 + u3));
    double e23 = norm(p2 - p3) * exp(0.5 * (u2 + u3));

    EXPECT_FLOAT_EQ(e01, norm(p[0] - p[1]));
    EXPECT_FLOAT_EQ(e02, norm(p[0] - p[2]));
    EXPECT_FLOAT_EQ(e03, norm(p[0] - p[3]));
    EXPECT_FLOAT_EQ(e12, norm(p[1] - p[2]));
    EXPECT_FLOAT_EQ(e13, norm(p[1] - p[3]));
    EXPECT_FLOAT_EQ(e23, norm(p[2] - p[3]));
}

TEST_F(TetTest, gradient) {
    Vector3 v{1, 2, 3};
    std::vector<Vector3> pos;
    std::vector<double> u;
    for (size_t i = 0; i < 4; ++i) {
        pos.emplace_back(tetMesh->vertices[i].position);
        u.emplace_back(dot(v, pos[i]));
    }

    Vector3 otherV = CompArch::grad(u, pos);

    EXPECT_VEC3_NEAR(v, otherV, 1e-8);
}

TEST_F(TetTest, divSumsToZero) {
    Vector3 X{1, 2, 3};
    std::vector<Vector3> p;
    for (size_t i = 0; i < 4; ++i) {
        p.emplace_back(tetMesh->vertices[i].position);
    }

    std::vector<double> divX = CompArch::div(X, p);

    EXPECT_NEAR(divX[0] + divX[1] + divX[2] + divX[3], 0, 1e-8);
}

TEST_F(TetTest, divSymmetric) {
    Vector3 X{0, 0, 1};
    std::vector<Vector3> p;
    p.emplace_back(Vector3{1, 0, 0});
    p.emplace_back(Vector3{cos(2 * PI / 3), sin(2 * PI / 3), 0});
    p.emplace_back(Vector3{cos(2 * PI / 3), -sin(2 * PI / 3), 0});
    p.emplace_back(Vector3{0, 0, 1});

    std::vector<double> divX = CompArch::div(X, p);

    EXPECT_NEAR(divX[0], divX[1], 1e-8);
    EXPECT_NEAR(divX[1], divX[2], 1e-8);
}
