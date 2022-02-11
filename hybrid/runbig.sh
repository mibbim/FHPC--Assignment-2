#!/bin/bash

# mpirun -np 2 ./nov_hybrid_kd_tree.x 100000000 4 > tmp.dot
mpirun -np 2 ./nov_hybrid_kd_tree.x 200000000 1 > tmp.dot
dot -Tsvg tmp.dot > tree.svg