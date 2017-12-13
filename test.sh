#!/bin/bash

BASEDIR=$(dirname $(readlink -f %0))

cd $BASEDIR/out/oil
. test.sh
cd $BASEDIR/out/resize
. test.sh
cd $BASEDIR/out/swirl
. test.sh
cd $BASEDIR
