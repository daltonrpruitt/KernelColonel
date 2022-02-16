#!/bin/bash
ml cmake
ml cuda 

mkdir output
mkdir -p build/debug

cd build/debug
cmake -B . -S ../..
make

echo "Run './build/debug/ctx_driver' to test"