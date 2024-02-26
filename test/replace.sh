#!/bin/bash

cp job.json ./BondOrderAnalysis/dpmd_run_script
cp job.json ./AvalancheDynamics/dpmd_run_script/equ

cp job.json ./AvalancheDynamics/dpmd_run_script/add_Ef

echo "All job.json files have been replaced."
