#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh

REPO_PATH=/mnt/WDRed4T/yibo/multi_task_NER/proposed_model
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/mnt/WDRed4T/yibo/multi_task_NER/outputs/multi_ner/0101/ace2004/large_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen128_bz2_newatt_lstm_gab4
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=19.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
GPU_ID=1

python3 ${REPO_PATH}/evaluate/multi_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID}