#!/bin/bash
protocol="dns"  # "http", "tls"
epoch=(250)
bsz=(8) #16 32 64 128
dataset="selfdeploy_24_25_2week" #merge_month_2_4_2week

for ep in "${epoch[@]}"; do
    for bs in "${bsz[@]}"; do
        num_train_iter=$((ep * 100))  
        save_name="iomatch_${dataset}_${protocol}_extra_ep${ep}_bs${bs}"
        load_path="./saved_models/openset_cv/iomatch_${dataset}_${protocol}_extra_ep${ep}_bs${bs}/latest_model.pth"
        nohup python train.py \
        --c config/openset_cv/iomatch/iomatch_exp_${dataset}_${protocol}.yaml \
        --epoch "$ep" \
        -bsz "$bs" \
        --num_train_iter "$num_train_iter" \
        -sn "$save_name" \
        --load_path "$load_path" \
        > "logs/${protocol}/iomatch_${dataset}_ep${ep}_bs${bs}.log" 2>&1 &
        echo "training start! ${dataset} epoch=$ep, batch_size=$bs"
    done
done
