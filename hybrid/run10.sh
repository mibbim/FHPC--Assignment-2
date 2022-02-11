#!/bin/bash

mpirun -np 2 ./hybrid_kd_tree.x 10 2 > tmp.dot
dot -Tpng tmp.dot > tree.png