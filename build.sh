#!/bin/bash
module load cmake
module load gcc
module load cuda

dirs="output build/debug"
for d in ${dirs[@]}; do 
    if test -d "$d"; then
        echo "$d exists.";
    else  
        mkdir -vp $d;
    fi
done

cd build/debug ;
if test ! -f "../CMakeCache.txt"; then
    cmake -B . -S ../.. ;
fi 

make

echo "Run './build/debug/deviceQuery' to output device properties"
echo "Run './build/debug/main' to test"

