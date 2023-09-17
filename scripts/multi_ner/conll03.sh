#!/usr/bin/env bash
# -*- coding: utf-8 -*-


TIME=1217
# FILE=conll03_cased_large
REPO_PATH=/mnt/WDRed4T/yibo/multi_task_NER/proposed_model
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/mnt/WDRed4T/yibo/multi_task_NER/proposed_model/data/conll03
BERT_DIR=bert-base-uncased
# OUTPUT_BASE=/userhome/xiaoya/outputs

BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=2e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=200
MAX_NORM=1.0
INTER_HIDDEN=2048


BATCH=1
PREC=16
VAL_CHECK=0.2
GRAD_ACC=2
MAX_EPOCH=20
LR_SCHEDULER=polydecay

LR_MINI=3e-7
WEIGHT_DECAY=0.01
OPTIM=torch.adam
SPAN_CAND=pred_and_gold


# OUTPUT_DIR=${OUTPUT_BASE}/mrc_ner/${TIME}/${FILE}_cased_large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
OUTPUT_DIR=/mnt/WDRed4T/yibo/multi_task_NER/outputs/multi_ner/${TIME}/conll03/large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}_bz${BATCH_SIZE}_tower0_noatt_diff
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0,1 python ${REPO_PATH}/multi_train_light.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="2" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--lr_mini ${LR_MINI}

