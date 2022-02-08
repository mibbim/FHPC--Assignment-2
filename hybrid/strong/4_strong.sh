#!/bin/bash
 
#PBS -q dssc
#PBS -l nodes=4:ppn=24
#PBS -l walltime=1:00:00
cd $PBS_O_WORKDIR

for N in 4, 8; do
for n in 1 2 4 8 12; do
    mpirun --map-by socket -np ${N} ../nov_hybrid_kd_tree.x 100000000 ${n} > tree_4_${n}.dot 2>>time_4.out
done
done