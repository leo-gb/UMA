#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34001 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34101 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34201 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34301 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34401 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34501 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34601 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=34701 train.py \
--distributed \
--task_name 'e13_baseline_r18_preFalse_step_v2' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-9_v2/e13_baseline_r18_preFalse_step_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/train_attrbute_9.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-9_v2/valid_attrbute_9.txt' \
--valid > e13_baseline_r18_preFalse_step_v2_log_07.txt &

