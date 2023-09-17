#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_query_ner.py

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from modeling.character_bert import CharacterBertModel

from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss

from models.classifier import MultiNonLinearClassifier
from models.model_config import BertQueryNerConfig
# from text_encoder import TextEncoder

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

class MMBertQueryNER(nn.Module):
    def __init__(self, 
                 mrc_dropout, 
                 classifier_intermediate_hidden_size,
                 tower_dims=[768],
                 drop_prob=[0.1]):
        super(MMBertQueryNER, self).__init__()

        self.lstm = nn.LSTM(input_size=768*2,
                            hidden_size=768 // 2,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.2,
                            batch_first=True)

        # Load some pre-trained CharacterBERT and BERT
        # self.character_model = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/')
        # self.bert = BertModel(config)
        # self.span_loss_candidates = "pred_and_gold"
        # self.bce_loss = BCEWithLogitsLoss(reduction="none")

        # tower -- info layer -- attention layer -- start/end/span
        # self.tower_input_size = config.hidden_size 

        # self.tower1 = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.attention1 = Attention(tower_dims[-1])
        # self.info1 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs1 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs1 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding1 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.tower2 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention2 = Attention(tower_dims[-1])
        # self.info2 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs2 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs2 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding2 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.tower3 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention3 = Attention(tower_dims[-1])
        # self.info3 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs3 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs3 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding3 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.tower4 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention4 = Attention(tower_dims[-1])
        # self.info4 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs4 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs4 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding4 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.tower5 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention5 = Attention(tower_dims[-1])
        # self.info5 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs5 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs5 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding5 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.tower6 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention6 = Attention(tower_dims[-1])
        # self.info6 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs6 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs6 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding6 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.tower7 = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.attention7 = Attention(tower_dims[-1])
        # self.info7 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.start_outputs7 = nn.Linear(tower_dims[-1], 1)
        self.end_outputs7 = nn.Linear(tower_dims[-1], 1)
        self.span_embedding7 = MultiNonLinearClassifier(tower_dims[-1] * 2, 1, mrc_dropout, intermediate_hidden_size=classifier_intermediate_hidden_size)

        # self.hidden_size = 768

        # self.init_weights()

    def forward(self, embeddings):
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
        # put samples with the same question togather
        # stack and obtain bert embedding for each question batch
        batch_size = 1
        batch_size_num_class, seq_len, hidden_dim = embeddings.shape
        # print("embeddings", embeddings.shape) # [batch_size*num_class, seq_len, 1536]
        lstm_output = self.lstm(embeddings)[0] 
        # print(lstm_output.shape)  # [batch_size*num_class, seq_len, 768]

        start_logits = []
        end_logits = []
        span_logits = []

        tower1 = lstm_output[0:batch_size, :, :] # [batch_size, seq_len, 768]
        tower2 = lstm_output[batch_size:2*batch_size, :, :]
        tower3 = lstm_output[2*batch_size:3*batch_size, :, :]
        tower4 = lstm_output[3*batch_size:4*batch_size, :, :]
        tower5 = lstm_output[4*batch_size:5*batch_size, :, :]
        tower6 = lstm_output[5*batch_size:6*batch_size, :, :]
        tower7 = lstm_output[6*batch_size:7*batch_size, :, :]
        
        ait1 = self.attention1(torch.cat([tower1, tower2, tower3, tower4, tower5, tower6, tower7], 1)) # [batch_size, num_class*seq_len, 768] => [batch_size, num_class*seq_len, 32] 
        # print("ait1\n")
        # print(ait1.shape) # [num_class*batch_size, seq_len, 32]
        # ait1_0 = ait1[0:batch_size, :, :] # [batch_size, seq_len, 32]
        ait1_0 = ait1[:, :seq_len, :]
        # print(ait1_0.shape)
        # ait1_0 = tower1
        # print(ait1_0.shape)
        # print("+++", self.start_outputs1(ait1_0).shape, self.end_outputs1(ait1_0).shape) # [batch_size, seq_len, 1]
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
        
        # ait1_4 = ait1[4*batch_size:5*batch_size, :, :]
        ait1_4 = ait1[:, 4*seq_len:5*seq_len, :]
        # ait1_4 = tower5
        start_logits.append(self.start_outputs5(ait1_4).squeeze(-1))
        end_logits.append(self.end_outputs5(ait1_4).squeeze(-1))
        start_extend = ait1_4.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1_4.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding5(span_matrix).squeeze(-1))
        
        # ait1_5 = ait1[5*batch_size:6*batch_size, :, :]
        ait1_5 = ait1[:, 5*seq_len:6*seq_len, :]
        # ait1_5 = tower6
        start_logits.append(self.start_outputs6(ait1_5).squeeze(-1))
        end_logits.append(self.end_outputs6(ait1_5).squeeze(-1))
        start_extend = ait1_5.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1_5.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding6(span_matrix).squeeze(-1))
        
        # ait1_6 = ait1[6*batch_size:7*batch_size, :, :]
        ait1_6 = ait1[:, 6*seq_len:7*seq_len, :]
        # ait1_6 = tower7
        start_logits.append(self.start_outputs7(ait1_6).squeeze(-1))
        end_logits.append(self.end_outputs7(ait1_6).squeeze(-1))
        start_extend = ait1_6.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait1_6.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding7(span_matrix).squeeze(-1))

        new_start_logits = []
        new_end_logits = []
        new_span_logits = []
        # print(len(start_logits))
        # print(start_logits[0].shape, end_logits[0].shape, span_logits[0].shape)
        for i in range(batch_size):
            for j in range(7):
                # print("start_logits", start_logits[j].shape)
                new_start_logits.append(start_logits[j][i]) 
                new_end_logits.append(end_logits[j][i])
                new_span_logits.append(span_logits[j][i])
        new_start_logits = torch.stack(new_start_logits) # [batch_size*num_class, seq_len]
        new_end_logits = torch.stack(new_end_logits)
        new_span_logits = torch.stack(new_span_logits)
        # print(new_start_logits.shape)

        # return new_start_logits, new_end_logits, new_span_logits, a
        return new_start_logits, new_end_logits, new_span_logits
