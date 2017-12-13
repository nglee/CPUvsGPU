#!/bin/sh

SEP="==================================================="

echo $SEP
echo "running oil_CPU"
echo $SEP
./oil_CPU

echo $SEP
echo "running oil_CPU_OMP"
echo $SEP
./oil_CPU_OMP

echo $SEP
echo "running oil_CUDA"
echo $SEP
./oil_CUDA

echo $SEP
echo "running oil_CUDA_SHARED"
echo $SEP
./oil_CUDA_SHARED

echo $SEP
echo "running oil_CUDA_SECOND"
echo $SEP
./oil_CUDA_SECOND

echo $SEP
echo "diff output images"
echo $SEP
for i in $(seq -f "%02g" 0 7)
do
    diff oil_"$i"_CPU.jpg oil_"$i"_CPU_OMP.jpg
    diff oil_"$i"_CPU.jpg oil_"$i"_CUDA.jpg
    diff oil_"$i"_CPU.jpg oil_"$i"_CUDA_SHARED.jpg
    diff oil_"$i"_CPU.jpg oil_"$i"_CUDA_SECOND.jpg
done
