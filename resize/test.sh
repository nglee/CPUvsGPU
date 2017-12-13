#!/bin/sh

SEP="==================================================="

echo $SEP
echo "running resize_CPU"
echo $SEP
./resize_CPU

echo $SEP
echo "running resize_CUDA"
echo $SEP
./resize_CUDA

