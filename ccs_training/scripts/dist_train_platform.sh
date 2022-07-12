#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

# 性感手办
nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36000 train.py \
--distributed \
--task_name 'e01_resnet50_baseline_v2' \
--cfg_path 'train_cfg_platform' \
--mode 'train' \
--gpus '0' \
--train_ratio 100 \
--delimiter '$$||$$' \
--round '1' \
--log_dir    'leogb/causal_logs/task_xgsb/e01_resnet50_baseline_v2/' \
--train_file 'leogb/causal_logs/task_xgsb/xgsb_train_data.txt' \
--valid_file 'leogb/causal_logs/task_xgsb/xgsb_valid_data.txt' \
--load_data_to_memory \
--valid > task_xgsb-e01_resnet50_baseline_v2.txt &

# # 内裤315
# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36000 train.py \
# --distributed \
# --task_name 'e01_resnet50_imagenetpretrained_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 100 \
# --delimiter '$$||$$' \
# --round '1' \
# --log_dir    'leogb/causal_logs/task_315/e01_resnet50_imagenetpretrained_v1/' \
# --train_file 'leogb/causal_logs/task_315/train_data_labeled_train.txt' \
# --valid_file 'leogb/causal_logs/task_315/train_data_labeled_val.txt' \
# --valid > e01_resnet50_imagenetpretrained_v1-task_315.txt &

# # 文胸320
# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36002 train.py \
# --distributed \
# --task_name 'e01_resnet50_imagenetpretrained_v1' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 100 \
# --delimiter '$$||$$' \
# --round '1' \
# --log_dir    'leogb/causal_logs/task_320/e01_resnet50_imagenetpretrained_v1/' \
# --train_file 'leogb/causal_logs/task_320/train_data_labeled_train.txt' \
# --valid_file 'leogb/causal_logs/task_320/train_data_labeled_val.txt' \
# --valid > e01_resnet50_imagenetpretrained_v1-task_320.txt &

# # 云朵包：310
# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36004 train.py \
# --distributed \
# --task_name 'e01_resnet50_imagenetpretrained_v1' \
# --mode 'train' \
# --gpus '4' \
# --train_ratio 100 \
# --delimiter '$$||$$' \
# --round '1' \
# --log_dir    'leogb/causal_logs/task_310/e01_resnet50_imagenetpretrained_v1/' \
# --train_file 'leogb/causal_logs/task_310/train_data_labeled_train.txt' \
# --valid_file 'leogb/causal_logs/task_310/train_data_labeled_val.txt' \
# --valid > e01_resnet50_imagenetpretrained_v1-task_310.txt &

# # 无人机：307
# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36006 train.py \
# --distributed \
# --task_name 'e01_resnet50_imagenetpretrained_v1' \
# --mode 'train' \
# --gpus '6' \
# --train_ratio 100 \
# --delimiter '$$||$$' \
# --round '1' \
# --log_dir    'leogb/causal_logs/task_307/e01_resnet50_imagenetpretrained_v1/' \
# --train_file 'leogb/causal_logs/task_307/train_data_labeled_train.txt' \
# --valid_file 'leogb/causal_logs/task_307/train_data_labeled_val.txt' \
# --valid > e01_resnet50_imagenetpretrained_v1-task_307.txt &
