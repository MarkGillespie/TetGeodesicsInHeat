#!/bin/bash

mkdir deps
cd deps
git submodule add https://github.com/nmwsharp/polyscope.git
git submodule add -b v1 https://github.com/nmwsharp/geometry-central.git
git submodule add https://github.com/google/googletest
git submodule update --init --recursive
cd ..
mkdir build
cd build
cmake ..

cd ../src
mkdir build
cd build
ln -s ../../build/compile_commands.json compile_commands.json
