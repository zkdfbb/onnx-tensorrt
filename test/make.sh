#!/usr/bin/env bash
# -*- coding: utf-8 -*-

##############################################
#
#  Author:
#  Email: @nio.com
#  Last modified: 2020-10-02 14:00:37
#
##############################################

test -e build && rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_62=on
make -j8
cd ..
