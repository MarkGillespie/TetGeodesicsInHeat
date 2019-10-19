#include "test_utils.h"
#include "tet.h"

class TetTest : public ::testing::Test {
 public:
  static CompArch::TetMesh* tetMesh;

 protected:
  static void SetUpTestSuite() {
    std::vector<Vector3> positions;
    positions.emplace_back(Vector3{0, 0, 0});
    positions.emplace_back(Vector3{1, 0, 0});
    positions.emplace_back(Vector3{0, 1, 0});
    positions.emplace_back(Vector3{0, 0, 1});
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
  double simpleVol = dot(v3, cross(v1 - v0, v2 - v0));

  EXPECT_FLOAT_EQ(vol, simpleVol);
}
