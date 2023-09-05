#!/bin/bash

# for ipc in {1..20}; do
#     echo "Running experiment with IPC=$ipc"
#     python distill.py --ipc $ipc --distillation_step 0 --syn_steps 20 --syn_init_method real --verbose
# done


for ((N=1; N<=100; N+=10)); do
    echo "Running experiment with N=$N"
    python distill.py --ipc 2 --distillation_step 0 --syn_steps $N --syn_init_method real --verbose
done

