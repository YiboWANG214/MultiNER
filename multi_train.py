#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: multi_train.py

import sys
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from models.multi_model import MMBertQueryNER
from models.model_config import BertQueryNerConfig

from datasets.collate_functions import collate_to_max_length

from metrics.query_span_f1 import QuerySpanF1
import logging
import argparse

# super parameter
batch_size = 1
embedding_size = 5
learning_rate = 5e-5
total_epoch = 10
earlystop_epoch = 1
DATA_DIR = "/mnt/WDRed4T/yibo/multi_task_NER/proposed_model/data/ace2004"

span_f1 = QuerySpanF1(False)
result_logger = logging.getLogger(__name__)
result_logger.setLevel(logging.INFO)
bert_config_dir = "bert-base-uncased"
bert_dropout = 0.1
mm_dropout = 0.3
classifier_act_func = "gelu"
classifier_intermediate_hidden_size = 1536


def train():
  with open(os.path.join(DATA_DIR, f"embedding.train"), 'rb') as train_file:
    train_dataset = pickle.load(train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=7*batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_to_max_length
    )
  with open(os.path.join(DATA_DIR, f"embedding.dev"), 'rb') as dev_file:
    dev_dataset = pickle.load(dev_file)
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=7*batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_to_max_length
    )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  bert_config = BertQueryNerConfig.from_pretrained(bert_config_dir,
                                                    hidden_dropout_prob=bert_dropout,
                                                    attention_probs_dropout_prob=bert_dropout,
                                                    mm_dropout=mm_dropout,
                                                    classifier_act_func = classifier_act_func,
                                                    classifier_intermediate_hidden_size=classifier_intermediate_hidden_size)

  model = MMBertQueryNER.from_pretrained(
                                         bert_config_dir,
                                         config=bert_config,
                                         ).to(device)

  weight_start = torch.nn.parameter.Parameter(torch.tensor(1.0).to(device))
  weight_end = torch.nn.parameter.Parameter(torch.tensor(1.0).to(device))
  weight_span = torch.nn.parameter.Parameter(torch.tensor(1.0).to(device))
  no_decay = ["bias", "LayerNorm.weight"]
  print("\n\n\n")
  for name, param in model.named_parameters():
    print(name)
  print("\n\n\n")
  optimizer_grouped_parameters = [
      {
          "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
          "weight_decay": 0.01,
      },
      {
          "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
          "weight_decay": 0.0,
      },
      # {
      #     'params': [weight_start, weight_end, weight_span], 
      #     'weight_decay': 0.01,
      # }, 
  ]
  optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                lr=learning_rate,
                                eps=1e-5,
                                weight_decay=0.01)
  model.to(device)
  best_acc = 0.0
  earystop_count = 0
  best_epoch = 0
  for epoch in range(total_epoch):
    the_total_loss = 0.
    nb_sample = 0
    # train
    model.train()
    for step, batch in enumerate(train_dataloader):
      tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
      attention_mask = (tokens != 0).long()
      # print(tokens.shape, attention_mask.shape, token_type_ids.shape)
      start_logits, end_logits, span_logits = model(tokens.to(device), attention_mask.to(device), token_type_ids.to(device))

      start_loss, end_loss, match_loss = model.compute_loss(start_logits=start_logits,
                                                      end_logits=end_logits,
                                                      span_logits=span_logits,
                                                      start_labels=start_labels.to(device),
                                                      end_labels=end_labels.to(device),
                                                      match_labels=match_labels.to(device),
                                                      start_label_mask=start_label_mask.to(device),
                                                      end_label_mask=end_label_mask.to(device)
                                                      )
      total_loss = weight_start * start_loss + weight_end * end_loss + weight_span * match_loss

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      the_total_loss += total_loss.cpu().detach().numpy()
      if step % 200 == 0:
        print('Train loss on step %d: %.6f' %
              ((step + 1), the_total_loss / (step + 1)))

    # validation
    print("start validation...")
    model.eval()
    eval_span_f1 = []
    eval_loss = []
    for batch in enumerate(dev_dataloader):
      tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
      attention_mask = (tokens != 0).long()
      start_logits, end_logits, span_logits = model(tokens, attention_mask, token_type_ids)

      start_loss, end_loss, match_loss = model.compute_loss(start_logits=start_logits,
                                                      end_logits=end_logits,
                                                      span_logits=span_logits,
                                                      start_labels=start_labels,
                                                      end_labels=end_labels,
                                                      match_labels=match_labels,
                                                      start_label_mask=start_label_mask,
                                                      end_label_mask=end_label_mask
                                                      )
      total_loss = weight_start * start_loss + weight_end * end_loss + weight_span * match_loss
      eval_loss.append(total_loss)
      start_preds, end_preds = start_logits > 0, end_logits > 0
      span_f1_stats = span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                              start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                              match_labels=match_labels)
      eval_span_f1.append(span_f1_stats)

    avg_loss = torch.mean(eval_loss)
    tensorboard_logs = {'val_loss': avg_loss}
    all_counts = torch.stack(eval_span_f1).view(-1, 3).sum(0)
    span_tp, span_fp, span_fn = all_counts
    span_recall = span_tp / (span_tp + span_fn + 1e-10)
    span_precision = span_tp / (span_tp + span_fp + 1e-10)
    span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
    tensorboard_logs[f"span_precision"] = span_precision
    tensorboard_logs[f"span_recall"] = span_recall
    tensorboard_logs[f"span_f1"] = span_f1
    result_logger.info(f"EVAL INFO -> current_epoch is: {epoch}, current_global_step is: {step} ")
    result_logger.info(f"EVAL INFO -> valid_f1 is: {span_f1}; precision: {span_precision}, recall: {span_recall}.")

    return {'val_loss': avg_loss, 'log': tensorboard_logs}


# def test():
#   print("Start Test ...")
#   with open(os.path.join(DATA_DIR, f"embedding.test"), 'rb') as test_file:
#     test_dataset = pickle.load(test_file)
#     test_dataloader = DataLoader(
#         dataset=test_dataset,
#         batch_size=batch_size,
#         num_workers=0,
#         shuffle=False,
#         collate_fn=collate_to_max_length
#     )
#   model = AITM(vocabulary_size, 5)
#   model.load_state_dict(torch.load(model_file))
#   model.eval()
#   click_list = []
#   conversion_list = []
#   click_pred_list = []
#   conversion_pred_list = []
#   for i, batch in enumerate(test_loader):
#     if i % 1000:
#       sys.stdout.write("test step:{}\r".format(i))
#       sys.stdout.flush()
#     click, conversion, features = batch
#     with torch.no_grad():
#       click_pred, conversion_pred = model(features)
#     click_list.append(click)
#     conversion_list.append(conversion)
#     click_pred_list.append(click_pred)
#     conversion_pred_list.append(conversion_pred)
#   click_auc = cal_auc(click_list, click_pred_list)
#   conversion_auc = cal_auc(conversion_list, conversion_pred_list)
#   print("Test Resutt: click AUC: {} conversion AUC:{}".format(
#       click_auc, conversion_auc))
#   if not os.path.exists('saved_labels2'):
#     os.makedirs('./saved_labels2/')
#   file1 = open('./saved_labels2/AITM_true_click_label.pickle', 'wb')
#   pickle.dump(preprocess(click_list), file1)
#   file2 = open('./saved_labels2/AITM_true_click_prediction.pickle', 'wb')
#   pickle.dump(preprocess(click_pred_list), file2)
#   file3 = open('./saved_labels2/AITM_true_conversion_label.pickle', 'wb')
#   pickle.dump(preprocess(conversion_list), file3)
#   file4 = open('./saved_labels2/AITM_true_conversion_prediction.pickle', 'wb')
#   pickle.dump(preprocess(conversion_pred_list), file4)

def preprocess(label):
  label = torch.cat(label)
  label = label.detach().tolist()
  return label


def cal_auc(label: list, pred: list):
  label = torch.cat(label)
  pred = torch.cat(pred)
  label = label.detach().numpy()
  pred = pred.detach().numpy()
  auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
  return auc


if __name__ == "__main__":
  train()
  # test()
