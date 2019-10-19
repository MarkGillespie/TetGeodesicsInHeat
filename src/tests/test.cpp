#include "test_utils.h"

#include <gtest/gtest.h>

#include "ExampleTests.h"
#include "tetTest.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
