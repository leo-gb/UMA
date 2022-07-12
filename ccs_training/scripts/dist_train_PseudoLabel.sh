#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36000 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e19_r18_preFalse_step_PseudoLabel_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--OverwriteDAConfig \
--UMAx10_PLv1 \
--valid > e19_r18_preFalse_step_PseudoLabel_v1-Attr19log_01.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36001 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
--mode 'train' \
--gpus '1' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e19_r18_preFalse_step_PseudoLabel_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--OverwriteDAConfig \
--UMAx10_PLv2 \
--valid > e19_r18_preFalse_step_PseudoLabel_v2-Attr19log_02.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36002 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
--mode 'train' \
--gpus '2' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e19_r18_preFalse_step_PseudoLabel_v3/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--OverwriteDAConfig \
--UMAx10_PLv3 \
--valid > e19_r18_preFalse_step_PseudoLabel_v3-Attr19log_03.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36003 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
--mode 'train' \
--gpus '3' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e19_r18_preFalse_step_PseudoLabel_v4/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--OverwriteDAConfig \
--UMAx10_PLv4 \
--valid > e19_r18_preFalse_step_PseudoLabel_v4-Attr19log_04.txt &

# ----------

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36004 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e19_r18_preFalse_step_PseudoLabel_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10_PLv1 \
--valid > e19_r18_preFalse_step_PseudoLabel_v1-Attr32log_01.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36005 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
--mode 'train' \
--gpus '5' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e19_r18_preFalse_step_PseudoLabel_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10_PLv2 \
--valid > e19_r18_preFalse_step_PseudoLabel_v2-Attr32log_02.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36006 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
--mode 'train' \
--gpus '6' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e19_r18_preFalse_step_PseudoLabel_v3/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10_PLv3 \
--valid > e19_r18_preFalse_step_PseudoLabel_v3-Attr32log_03.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36007 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
--mode 'train' \
--gpus '7' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e19_r18_preFalse_step_PseudoLabel_v4/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--OverwriteDAConfig \
--UMAx10_PLv4 \
--valid > e19_r18_preFalse_step_PseudoLabel_v4-Attr32log_04.txt &

# ----------

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36008 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e19_r18_preFalse_step_PseudoLabel_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--OverwriteDAConfig \
--UMAx10_PLv1 \
--valid > e19_r18_preFalse_step_PseudoLabel_v1-Attr16log_01.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36009 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
--mode 'train' \
--gpus '1' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e19_r18_preFalse_step_PseudoLabel_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--OverwriteDAConfig \
--UMAx10_PLv2 \
--valid > e19_r18_preFalse_step_PseudoLabel_v2-Attr16log_02.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36010 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
--mode 'train' \
--gpus '2' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e19_r18_preFalse_step_PseudoLabel_v3/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--OverwriteDAConfig \
--UMAx10_PLv3 \
--valid > e19_r18_preFalse_step_PseudoLabel_v3-Attr16log_03.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36011 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
--mode 'train' \
--gpus '3' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e19_r18_preFalse_step_PseudoLabel_v4/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--OverwriteDAConfig \
--UMAx10_PLv4 \
--valid > e19_r18_preFalse_step_PseudoLabel_v4-Attr16log_04.txt &

# ----------

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36012 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e19_r18_preFalse_step_PseudoLabel_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--OverwriteDAConfig \
--UMAx10_PLv1 \
--valid > e19_r18_preFalse_step_PseudoLabel_v1-CA3log_01.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36013 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
--mode 'train' \
--gpus '5' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e19_r18_preFalse_step_PseudoLabel_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--OverwriteDAConfig \
--UMAx10_PLv2 \
--valid > e19_r18_preFalse_step_PseudoLabel_v2-CA3log_02.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36014 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
--mode 'train' \
--gpus '6' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e19_r18_preFalse_step_PseudoLabel_v3/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--OverwriteDAConfig \
--UMAx10_PLv3 \
--valid > e19_r18_preFalse_step_PseudoLabel_v3-CA3log_03.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36015 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
--mode 'train' \
--gpus '7' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e19_r18_preFalse_step_PseudoLabel_v4/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--OverwriteDAConfig \
--UMAx10_PLv4 \
--valid > e19_r18_preFalse_step_PseudoLabel_v4-CA3log_04.txt &

# ----------

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36016 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e19_r18_preFalse_step_PseudoLabel_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--OverwriteDAConfig \
--UMAx10_PLv1 \
--valid > e19_r18_preFalse_step_PseudoLabel_v1-CA5log_01.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36017 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v2' \
--mode 'train' \
--gpus '5' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e19_r18_preFalse_step_PseudoLabel_v2/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--OverwriteDAConfig \
--UMAx10_PLv2 \
--valid > e19_r18_preFalse_step_PseudoLabel_v2-CA5log_02.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36018 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v3' \
--mode 'train' \
--gpus '6' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e19_r18_preFalse_step_PseudoLabel_v3/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--OverwriteDAConfig \
--UMAx10_PLv3 \
--valid > e19_r18_preFalse_step_PseudoLabel_v3-CA5log_03.txt &

nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=36019 train.py \
--distributed \
--task_name 'e19_r18_preFalse_step_PseudoLabel_v4' \
--mode 'train' \
--gpus '7' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e19_r18_preFalse_step_PseudoLabel_v4/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--OverwriteDAConfig \
--UMAx10_PLv4 \
--valid > e19_r18_preFalse_step_PseudoLabel_v4-CA5log_04.txt &
