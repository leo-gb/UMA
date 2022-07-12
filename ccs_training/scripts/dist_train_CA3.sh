#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46001 train.py \
# --distributed \
# --task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
# --mode 'train' \
# --gpus '0' \
# --train_ratio 1 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
# --valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_00.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46101 train.py \
# --distributed \
# --task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
# --mode 'train' \
# --gpus '1' \
# --train_ratio 2 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
# --valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_01.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46201 train.py \
# --distributed \
# --task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
# --mode 'train' \
# --gpus '2' \
# --train_ratio 4 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
# --valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_02.txt &


# nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46301 train.py \
# --distributed \
# --task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
# --mode 'train' \
# --gpus '3' \
# --train_ratio 8 \
# --delimiter '||$$||' \
# --round '1' \
# --log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
# --train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
# --valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
# --valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46401 train.py \
--distributed \
--task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 3 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46501 train.py \
--distributed \
--task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 5 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46601 train.py \
--distributed \
--task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 6 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=46701 train.py \
--distributed \
--task_name 'e08_r18_preFalse_step_RandomErasing_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 7 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e08_r18_preFalse_step_RandomErasing_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e08_r18_preFalse_step_RandomErasing_v1_CA3log_07.txt &

