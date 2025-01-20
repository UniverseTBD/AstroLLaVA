#!/bin/bash
#PBS -q gpursaa
#PBS -j oe
#PBS -l walltime=01:00:00,mem=276GB
#PBS -l wd
#PBS -l ncpus=14
#PBS -l ngpus=1
#PBS -l storage=gdata/dk92
#

module load python3/3.10.0
export PYTHONPATH=`realpath ../.local/lib/`
source venv/bin/activate

