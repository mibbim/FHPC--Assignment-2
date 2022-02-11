#!/bin/bash

mpirun -np 4 ./hybrid_kd_tree.x 10 1 > tmp.dot
dot -Tsvg tmp.dot > tree.svg