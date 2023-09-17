#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_query_ner.py

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss

from models.classifier import MultiNonLinearClassifier
from models.model_config import BertQueryNerConfig

class Tower(nn.Module):
  def __init__(self,
               input_dim: int,
               dims=[128, 64, 16],
               drop_prob=[0.1, 0.3, 0.3]):
    super(Tower, self).__init__()
    self.dims = dims
    self.drop_prob = drop_prob
    self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                                # nn.Linear(dims[0], dims[1]), nn.ReLU(),
                                # nn.Dropout(drop_prob[1]),
                                # nn.Linear(dims[1], dims[2]), nn.ReLU(),
                                # nn.Dropout(drop_prob[2]),
                               )

  def forward(self, x):
    # print("1", x.shape) # [bz, seq_len, 768]
    # x = torch.flatten(x, start_dim=2)
    # print("2", x.shape) # [bz, seq_len, 768]
    x = self.layer(x)
    # print("3", x.shape) # [bz, seq_len, 32]
    return x

class Attention(nn.Module):
  def __init__(self, dim=7):
    super(Attention, self).__init__()
    # self.dim = dims
    self.q_layer = nn.Linear(dim, dim, bias=False)
    self.k_layer = nn.Linear(dim, dim, bias=False)
    self.v_layer = nn.Linear(dim, dim, bias=False)
    # self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs):
    # inputs = torch.flatten(inputs, end_dim=1)
    q = self.q_layer(inputs)
    k = self.k_layer(inputs)
    v = self.v_layer(inputs)
    a = torch.matmul(q, k.transpose(1, 2))
    # print("a", a.shape) # [num_class*batch_size, seq_len, seq_len]
    # [batch_size, num_class*seq_len, num_class*seq_len]
    outputs = torch.matmul(a, v)
    # return outputs, a
    return outputs

class MMBertQueryNER(BertPreTrainedModel):
    def __init__(self, 
                 config, 
                 tower_dims=[768],
                 drop_prob=[0.1]):
        super(MMBertQueryNER, self).__init__(config)
        self.bert = BertModel(config)
        # self.span_loss_candidates = "pred_and_gold"
        self.bce_loss = BCEWithLogitsLoss(reduction="none")

        # tower -- info layer -- attention layer -- start/end/span
        self.tower_input_size = config.hidden_size 

        # self.tower1 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention1 = Attention(tower_dims[-1])
        # self.info1 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs1 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs1 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding1 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        # self.tower2 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention2 = Attention(tower_dims[-1])
        # self.info2 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs2 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs2 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding2 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        # self.tower3 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention3 = Attention(tower_dims[-1])
        # self.info3 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs3 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs3 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding3 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        # self.tower4 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention4 = Attention(tower_dims[-1])
        # self.info4 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs4 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs4 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding4 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [batch_size, num_class, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        # bert_config_dir = "bert-base-uncased"
        # bert_dropout = 0.1
        # mm_dropout = 0.3
        # classifier_act_func = "gelu"
        # classifier_intermediate_hidden_size = 2048
        # bert_config = BertQueryNerConfig.from_pretrained(bert_config_dir,
        #                                             hidden_dropout_prob=bert_dropout,
        #                                             attention_probs_dropout_prob=bert_dropout,
        #                                             mm_dropout=mm_dropout,
        #                                             classifier_act_func = classifier_act_func,
        #                                             classifier_intermediate_hidden_size=classifier_intermediate_hidden_size)
        # bert = BertModel(bert_config).to('cuda')

        # put samples with the same question togather
        # stack and obtain bert embedding for each question batch
        bert_outputs = []
        # print("\n==========")
        # print(input_ids.shape)
        batch_size, num_class, seq_len = input_ids.shape
        for j in range(num_class):
            new_input_ids = []
            new_token_type_ids = []
            new_attention_mask = []
            for i in range(batch_size):
                new_input_ids.append(input_ids[i][j])
                new_token_type_ids.append(token_type_ids[i][j])
                new_attention_mask.append(attention_mask[i][j])
            tmp_input_ids = torch.stack(new_input_ids)
            tmp_token_type_ids = torch.stack(new_token_type_ids)
            tmp_attention_mask = torch.stack(new_attention_mask)
            # print(tmp_input_ids.shape)
            bert_output = self.bert(tmp_input_ids, token_type_ids=tmp_token_type_ids, attention_mask=tmp_attention_mask)
            bert_output = bert_output[0] # [batch_size, seq_len, 768]
            # print("bert_output", bert_output.shape)
            bert_outputs.append(bert_output)

        start_logits = []
        end_logits = []
        span_logits = []

        tower1 = bert_outputs[0] # [batch_size, seq_len, 768]
        tower2 = bert_outputs[1]
        tower3 = bert_outputs[2]
        tower4 = bert_outputs[3]

        # print("!", torch.cat([tower1, tower2, tower3, tower4, tower5, tower6, tower7], 1).shape) # [num_class, batch_size, seq_len, 768]
        # # ait1, a = self.attention1(torch.cat([tower1, tower2, tower3, tower4, tower5, tower6, tower7], 1)) # [num_class, batch_size, seq_len, 32]
        # ait1 = self.attention1(torch.cat([tower1, tower2, tower3, tower4, tower5, tower6, tower7], 1)) # [num_class, batch_size, seq_len, 768] => [num_class, batch_size, seq_len, 32]
        ait1 = self.attention1(torch.cat([tower1, tower2, tower3, tower4], 1)) # [batch_size, num_class*seq_len, 768] => [batch_size, num_class*seq_len, 32] 
        
        # print(ait1.shape) # [num_class*batch_size, seq_len, 32]
        # ait1_0 = ait1[0:batch_size, :, :] # [batch_size, seq_len, 32]
        ait1_0 = ait1[:, :seq_len, :]
        # ait1_0 = tower1
        # print("+++", self.start_outputs1(ait1_0).shape, self.end_outputs1(ait1_0).shape) # [batch_size, seq_len, 32]
        start_logits.append(self.start_outputs1(ait1_0).squeeze(-1)) # [batch_size, seq_len]
        # print(start_logits[0].shape)
        end_logits.append(self.end_outputs1(ait1_0).squeeze(-1))
        start_extend = ait1_0.unsqueeze(2).expand(-1, -1, seq_len, -1) # [batch_size, seq_len, seq_len, 32]
        end_extend = ait1_0.unsqueeze(1).expand(-1, seq_len, -1, -1) # [batch_size, seq_len, seq_len, 32]
        span_matrix = torch.cat([start_extend, end_extend], 3) # [batch_size, seq_len, seq_len, 64]
        span_logits.append(self.span_embedding1(span_matrix).squeeze(-1)) # [batch_size, seq_len, seq_len]
        # print("======", self.start_outputs1(ait1_0).shape, self.span_embedding1(span_matrix).shape)
        
        # ait1_1 = ait1[batch_size:2*batch_size, :, :]
        ait1_1 = ait1[:, seq_len:2*seq_len, :]
        # ait1_1 = tower2
        start_logits.append(self.start_outputs2(ait1_1).squeeze(-1))
        end_logits.append(self.end_outputs2(ait1_1).squeeze(-1))
        start_extend = ait1_1.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1_1.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding2(span_matrix).squeeze(-1))
        
        # ait1_2 = ait1[2*batch_size:3*batch_size, :, :]
        ait1_2 = ait1[:, 2*seq_len:3*seq_len, :]
        # ait1_2 = tower3
        start_logits.append(self.start_outputs3(ait1_2).squeeze(-1))
        end_logits.append(self.end_outputs3(ait1_2).squeeze(-1))
        start_extend = ait1_2.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1_2.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding3(span_matrix).squeeze(-1))
        
        # ait1_3 = ait1[3*batch_size:4*batch_size, :, :]
        ait1_3 = ait1[:, 3*seq_len:4*seq_len, :]
        # ait1_3 = tower4
        start_logits.append(self.start_outputs4(ait1_3).squeeze(-1))
        end_logits.append(self.end_outputs4(ait1_3).squeeze(-1))
        start_extend = ait1_3.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1_3.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding4(span_matrix).squeeze(-1))
        
        new_start_logits = []
        new_end_logits = []
        new_span_logits = []
        # print(len(start_logits))
        # print(start_logits[0].shape, end_logits[0].shape, span_logits[0].shape)
        for i in range(batch_size):
            for j in range(num_class):
                new_start_logits.append(start_logits[j][i]) 
                new_end_logits.append(end_logits[j][i])
                new_span_logits.append(span_logits[j][i])
        new_start_logits = torch.stack(new_start_logits) # [batch_size*num_class, seq_len]
        new_end_logits = torch.stack(new_end_logits)
        new_span_logits = torch.stack(new_span_logits)
        # print(new_start_logits.shape)

        # return new_start_logits, new_end_logits, new_span_logits, a
        return new_start_logits, new_end_logits, new_span_logits
