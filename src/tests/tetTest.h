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

  void SetUp() override { }
};

CompArch::TetMesh* TetTest::tetMesh = nullptr;

TEST_F(TetTest, volumeFormulaTest) {
  CompArch::Tet t = tetMesh->tets[0];
  double vol = tetMesh->tetVolume(t);

  Vector3 v0 = tetMesh->vertices[t.verts[0]].position;
  Vector3 v1 = tetMesh->vertices[t.verts[1]].position;
  Vector3 v2 = tetMesh->vertices[t.verts[2]].position;
  Vector3 v3 = tetMesh->vertices[t.verts[3]].position;
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
