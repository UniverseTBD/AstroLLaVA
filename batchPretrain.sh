#!/bin/bash
#PBS -q gpursaa
#PBS -j oe
#PBS -l walltime=06:30:00,mem=270GB
#PBS -l wd
#PBS -l ncpus=56
#PBS -l ngpus=4
#PBS -M sharaf.zaman@anu.edu.au
#PBS -m ae
#

module load python3/3.10.0
source venv/bin/activate
export PYTHONPATH=`realpath ../.local/lib/`
export PATH="/home/444/sz7583/.local/bin:$PATH"

/scratch/dg97/sz7583/LLaVA/scripts/v1_5/pretrain.sh
