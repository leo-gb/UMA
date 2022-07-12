#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=30001 train.py \
--distributed \
--task_name 'e00_tmptest' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e00_tmptest/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e00_tmptest_19log_00.txt &