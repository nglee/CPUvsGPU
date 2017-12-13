#!/bin/sh

SEP="==================================================="

echo $SEP
echo "running swirl_CPU"
echo $SEP
./swirl_CPU

echo $SEP
echo "running swirl_CPU_OMP"
echo $SEP
./swirl_CPU_OMP

echo $SEP
echo "running swirl_CUDA"
echo $SEP
./swirl_CUDA

echo $SEP
echo "diff output images"
echo $SEP
for i in $(seq -f "%02g" 0 7)
do
    diff swirl_"$i"_CPU.jpg swirl_"$i"_CPU_OMP.jpg
    diff swirl_"$i"_CPU.jpg swirl_"$i"_CUDA.jpg
done
