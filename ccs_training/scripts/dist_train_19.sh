#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30001 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_19log_00.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30101 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '1' \
# --train_ratio 2 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_19log_01.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30201 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 4 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_19log_02.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30301 train.py \
# --distributed \
# --task_name 'e18_r18_preFalse_step_OurX10_v1' \
# --mode 'train' \
# --gpus '3' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
# --valid > e18_r18_preFalse_step_OurX10_v1_19log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30401 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_19log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30501 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_19log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30601 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_19log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30701 train.py \
--distributed \
--task_name 'e18_r18_preFalse_step_OurX10_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e18_r18_preFalse_step_OurX10_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e18_r18_preFalse_step_OurX10_v1_19log_07.txt &

