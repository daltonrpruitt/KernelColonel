#!/bin/bash
module load cmake
module load gcc
module load cuda

dirs="output build/debug matrices"
for d in ${dirs[@]}; do 
    if test -d "$d"; then
        echo "$d exists.";
    else  
        mkdir -vp $d;
    fi
done

ml python
python3 download_matrices.py

cd build/debug ;
if test ! -f "../CMakeCache.txt"; then
    cmake -B . -S ../.. ;
fi 

make


echo "Run './build/debug/main' or './build/debug/spmv' to test"
