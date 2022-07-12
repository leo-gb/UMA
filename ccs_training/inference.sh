CKPT_PATH="leogb/causal_logs/CelebAMask-HQ-Attr9/e01_r18_M1N5_M2N5_PN_new/e01_r18_M1N5_M2N5_PN_new_0001_1/ckpt_ep26.pth"

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM3L4.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM3L4_prelabel_095.json" \
--gpus            '0' \
--threshold       0.95 > log-Attr9-editingM3L4.log &
 
nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM3L6.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM3L6_prelabel_095.json" \
--gpus            '1' \
--threshold       0.95 > log-Attr9-editingM3L6.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM3L8.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM3L8_prelabel_095.json" \
--gpus            '2' \
--threshold       0.95 > log-Attr9-editingM3L8.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM3L10.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM3L10_prelabel_095.json" \
--gpus            '3' \
--threshold       0.95 > log-Attr9-editingM3L10.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM3L12.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM3L12_prelabel_095.json" \
--gpus            '4' \
--threshold       0.95 > log-Attr9-editingM3L12.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop4.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop4_prelabel_095.json" \
--gpus            '5' \
--threshold       0.95 > log-Attr9-editingM4Ltop4.log &
 
nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop6.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop6_prelabel_095.json" \
--gpus            '6' \
--threshold       0.95 > log-Attr9-editingM4Ltop6.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop8.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop8_prelabel_095.json" \
--gpus            '7' \
--threshold       0.95 > log-Attr9-editingM4Ltop8.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop10.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop10_prelabel_095.json" \
--gpus            '0' \
--threshold       0.95 > log-Attr9-editingM4Ltop10.log &

nohup python -u inference.py \
--checkpoint_path ${CKPT_PATH} \
--input_file      "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop12.json" \
--save_file       "leogb/causal/CelebAMask-HQ-Attr9/editingM4Ltop12_prelabel_095.json" \
--gpus            '1' \
--threshold       0.95 > log-Attr9-editingM4Ltop12.log &
