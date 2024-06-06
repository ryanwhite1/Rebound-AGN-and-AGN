#!/bin/bash

MASS=8
FEDD=0.2
ALPH=0.01

python3 generate_SG_disk.py $MASS $FEDD $ALPH

make
./rebound << EOF
$MASS 
$FEDD 
$ALPH
EOF

python3 plot_sim.py $MASS $FEDD $ALPH