#!/bin/bash

PREFIX="tf2_test"
POOLING="stride"
DIMS=32
BATCH=16
NORMALIZATION="batch"
V_NORMALIZATION="global"
INITIALIZER="glorot_uniform"
L2=0.01


for iter in 0 1 2 3 4 5 6 7 8 9; do
  python train.py --dims=${DIMS} --session=${PREFIX}_d${DIMS}_init_${INITIALIZER}_b${BATCH}_pooling_${POOLING}_norm_${NORMALIZATION}_vnorm_${V_NORMALIZATION}_l2_${L2}_iter_${iter} --pooling=${POOLING} --normalization=${NORMALIZATION} --v-normalization=${V_NORMALIZATION} --l2=${L2} --batch=${BATCH} --kernel-initializer=${INITIALIZER} --random-seed=${iter}
done
