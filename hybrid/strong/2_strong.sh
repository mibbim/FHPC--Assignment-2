#!/bin/bash
 
#PBS -q dssc
#PBS -l nodes=2:ppn=24
#PBS -l walltime=1:00:00

module load openmpi
cd $PBS_O_WORKDIR

for N in 2 4; do
for n in 1 2 4 8 12; do
    mpirun --map-by socket -np ${N} ../nov_hybrid_kd_tree.x 100000000 ${n} 2>>time_${N}.out
done
done