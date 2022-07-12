# import psutil
# while psutil.pid_exists(84493):
#     print('{} is running, waiting...'.format(84493))
#     time.sleep(100)

#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40101 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40201 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40301 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40401 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40501 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40601 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=40701 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr19/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/train_attrbute_19.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr19/valid_attrbute_19.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_19log_07.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43001 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43101 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43201 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43301 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43401 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43501 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43601 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=43701 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/train_attrbute_32.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr32/valid_attrbute_32.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_32log_07.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45001 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45101 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45201 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45301 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45401 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45501 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45601 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=45701 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-Attr16/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/train_attrbute_16.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-Attr16/valid_attrbute_16.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_16log_07.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44001 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44101 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44201 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44301 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44401 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44501 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44601 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=44701 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/train_attrbute_CA3.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr3-19-32/valid_attrbute_CA3.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA3log_07.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49001 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '0' \
--train_ratio 1 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_00.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49101 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '1' \
--train_ratio 2 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_01.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49201 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '2' \
--train_ratio 4 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_02.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49301 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '3' \
--train_ratio 8 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_03.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49401 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '4' \
--train_ratio 16 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_04.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49501 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '5' \
--train_ratio 32 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_05.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49601 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '6' \
--train_ratio 64 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_06.txt &


nohup $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=49701 train.py \
--distributed \
--task_name 'e16_r18_preFalse_step_StyleMix_v1' \
--mode 'train' \
--gpus '7' \
--train_ratio 128 \
--delimiter '||$$||' \
--round '1' \
--log_dir    'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/e16_r18_preFalse_step_StyleMix_v1/' \
--train_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/train_attrbute_CA5.txt' \
--valid_file 'leogb/causal_logs/CelebAMask-HQ-ComAttr5-21-9/valid_attrbute_CA5.txt' \
--valid > e16_r18_preFalse_step_StyleMix_v1_CA5log_07.txt &

