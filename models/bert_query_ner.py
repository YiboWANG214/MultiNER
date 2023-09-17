#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_query_ner.py

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from models.classifier import MultiNonLinearClassifier

class Tower(nn.Module):
  def __init__(self,
               input_dim: int,
               dims=[128, 32],
               drop_prob=[0.1, 0.3]):
    super(Tower, self).__init__()
    self.dims = dims
    self.drop_prob = drop_prob
    self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                               nn.Linear(dims[0], dims[1]), nn.ReLU(),
                               nn.Dropout(drop_prob[1]),
                               nn.Linear(dims[1], dims[2]), nn.ReLU(),
                               nn.Dropout(drop_prob[2]))

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.layer(x)
    return x

class Attention(nn.Module):
  def __init__(self, dim=32):
    super(Attention, self).__init__()
    self.dim = dim
    self.q_layer = nn.Linear(dim, dim, bias=False)
    self.k_layer = nn.Linear(dim, dim, bias=False)
    self.v_layer = nn.Linear(dim, dim, bias=False)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs):
    Q = self.q_layer(inputs)
    K = self.k_layer(inputs)
    V = self.v_layer(inputs)
    a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
    a = self.softmax(a)
    outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
    return outputs

class BertQueryNER(BertPreTrainedModel):
    def __init__(self, 
                 config, 
                 tower_dims=[128, 64, 32],
                 drop_prob=[0.1, 0.3, 0.3]):
        super(BertQueryNER, self).__init__(config)
        self.bert = BertModel(config)

        # tower -- info layer -- attention layer -- start/end/span
        self.tower_input_size = config.hidden_size * 5
        self.tower1 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention1 = Attention(tower_dims[-1])
        self.info1 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs1 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs1 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding1 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.tower2 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention2 = Attention(tower_dims[-1])
        self.info2 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs2 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs2 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding2 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.tower3 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention3 = Attention(tower_dims[-1])
        self.info3 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs3 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs3 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding3 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.tower4 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention4 = Attention(tower_dims[-1])
        self.info4 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs4 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs4 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding4 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.tower5 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention5 = Attention(tower_dims[-1])
        self.info5 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs5 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs5 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding5 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.tower6 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention6 = Attention(tower_dims[-1])
        self.info6 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs6 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs6 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding6 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.tower7 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.start_outputs7 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs7 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding7 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, config.mrc_dropout, intermediate_hidden_size=config.classifier_intermediate_hidden_size)


        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = []
        end_logits = []
        span_logits = []

        tower1 = self.tower1(sequence_heatmap[0])
        start_logits.append(self.start_outputs(tower1).squeeze(-1))
        end_logits.append(self.end_outputs(tower1).squeeze(-1))
        start_extend = tower1.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = tower1.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))

        tower2 = self.tower2(sequence_heatmap[1])
        info1 = torch.unsqueeze(self.info1(tower1), 1)
        ait1 = self.attention1(torch.cat([tower2, info1], 1))
        start_logits.append(self.start_outputs(ait1).squeeze(-1))
        end_logits.append(self.end_outputs(ait1).squeeze(-1))
        start_extend = ait1.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))

        tower3 = self.tower3(sequence_heatmap[2])
        info2 = torch.unsqueeze(self.info2(tower2), 1)
        ait2 = self.attention2(torch.cat([tower3, info2], 1))
        start_logits.append(self.start_outputs(ait2).squeeze(-1))
        end_logits.append(self.end_outputs(ait2).squeeze(-1))
        start_extend = ait2.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait2.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))

        tower4 = self.tower4(sequence_heatmap[1])
        info3 = torch.unsqueeze(self.info3(tower3), 1)
        ait3 = self.attention1(torch.cat([tower4, info3], 1))
        start_logits.append(self.start_outputs(ait3).squeeze(-1))
        end_logits.append(self.end_outputs(ait3).squeeze(-1))
        start_extend = ait3.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait3.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))

        tower5 = self.tower5(sequence_heatmap[1])
        info4 = torch.unsqueeze(self.info4(tower4), 1)
        ait4 = self.attention1(torch.cat([tower5, info4], 1))
        start_logits.append(self.start_outputs(ait4).squeeze(-1))
        end_logits.append(self.end_outputs(ait4).squeeze(-1))
        start_extend = ait4.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait4.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))

        tower6 = self.tower6(sequence_heatmap[1])
        info5 = torch.unsqueeze(self.info5(tower5), 1)
        ait5 = self.attention1(torch.cat([tower6, info5], 1))
        start_logits.append(self.start_outputs(ait5).squeeze(-1))
        end_logits.append(self.end_outputs(ait5).squeeze(-1))
        start_extend = ait5.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait5.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))

        tower7 = self.tower7(sequence_heatmap[1])
        info6 = torch.unsqueeze(self.info6(tower6), 1)
        ait6 = self.attention1(torch.cat([tower7, info6], 1))
        start_logits.append(self.start_outputs(ait6).squeeze(-1))
        end_logits.append(self.end_outputs(ait6).squeeze(-1))
        start_extend = ait6.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait6.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding(span_matrix).squeeze(-1))


        return start_logits, end_logits, span_logits
