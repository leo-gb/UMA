#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38001 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38101 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38201 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 3 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38301 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38401 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 5 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38501 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 6 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38601 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 7 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=38701 train.py \
--distributed \
--task_name 'e23_r18_preFalse_step_OurXN_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e23_r18_preFalse_step_OurXN_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10 \
--valid > e23_r18_preFalse_step_OurXN_v1_32log_07.txt &

