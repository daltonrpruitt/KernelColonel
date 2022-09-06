#!/bin/bash
module load cmake
module load gcc
module load cuda

dirs="output build matrices"
for d in ${dirs[@]}; do 
    if test -d "$d"; then
        echo "$d exists.";
    else  
        mkdir -vp $d;
    fi
done

#ml python
#python3 download_matrices.py

cd build ;
if test ! -f "../CMakeCache.txt"; then
    cmake -B . -S .. ;
fi 

retval=make
if [[ retval -ne 0 ]]; then
   exit retval
fi

# Run Tests
cd ./tests
test_files=$(find -maxdepth 1 -executable -type f)
for f in $test_files;
do
    ./$f
done
