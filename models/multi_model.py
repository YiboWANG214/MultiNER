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
               dims=[128, 64, 32],
               drop_prob=[0.1, 0.3, 0.3]):
    super(Tower, self).__init__()
    self.dims = dims
    self.drop_prob = drop_prob
    self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                            #    nn.Linear(dims[0], dims[1]), nn.ReLU(),
                            #    nn.Dropout(drop_prob[1]),
                            #    nn.Linear(dims[1], dims[2]), nn.ReLU(),
                            #    nn.Dropout(drop_prob[2]),
                               )

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, start_dim=2)
    # print(x.shape)
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

class MMBertQueryNER(BertPreTrainedModel):
    def __init__(self, 
                 config, 
                 tower_dims=[32, 64, 32],
                 drop_prob=[0.1, 0.3, 0.3]):
        super(MMBertQueryNER, self).__init__(config)
        self.bert = BertModel(config).to('cuda')
        self.span_loss_candidates = "all"
        self.bce_loss = BCEWithLogitsLoss(reduction="none")

        # tower -- info layer -- attention layer -- start/end/span
        self.tower_input_size = config.hidden_size 

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
        self.attention7 = Attention(tower_dims[-1])
        self.info7 = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(), nn.Dropout(drop_prob[-1]))
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
        # bert_config_dir = "bert-base-uncased"
        # bert_dropout = 0.1
        # mm_dropout = 0.3
        # classifier_act_func = "gelu"
        # classifier_intermediate_hidden_size = 1536
        # bert_config = BertQueryNerConfig.from_pretrained(bert_config_dir,
        #                                             hidden_dropout_prob=bert_dropout,
        #                                             attention_probs_dropout_prob=bert_dropout,
        #                                             mm_dropout=mm_dropout,
        #                                             classifier_act_func = classifier_act_func,
        #                                             classifier_intermediate_hidden_size=classifier_intermediate_hidden_size)
        # bert = BertModel(bert_config).to('cuda')

        new_input_ids = [[]]*7
        new_token_type_ids = [[]]*7
        new_attention_mask = [[]]*7

        # put samples with the same question togather
        for i in range(len(input_ids)):
            new_input_ids[i%7].append(input_ids[i])
            new_token_type_ids[i%7].append(token_type_ids[i])
            new_attention_mask[i%7].append(attention_mask[i])

        # stack and obtain bert embedding for each question batch
        bert_outputs = []
        for i in range(7):
            input_ids = torch.stack(new_input_ids[i])
            token_type_ids = torch.stack(new_token_type_ids[i])
            attention_mask = torch.stack(new_attention_mask[i])
            # print(input_ids)
            bert_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
            bert_outputs.append(bert_output)

        # sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = bert_outputs[0].size()
        # seq_len = 32

        start_logits = []
        end_logits = []
        span_logits = []

        # print(bert_outputs[0].shape)
        # print(self.tower_input_size)
        tower1 = self.tower1(bert_outputs[0]) # bert_outputs[0]: [7, 43, 768] [batch_size, seq_len, hidden_size]
        # print("!", tower1.shape)
        tower2 = self.tower2(bert_outputs[1]) # tower1: [7, 43, 32] [batch_size, seq_len, 32]
        tower3 = self.tower3(bert_outputs[2])
        tower4 = self.tower4(bert_outputs[3])
        tower5 = self.tower5(bert_outputs[4])
        tower6 = self.tower6(bert_outputs[5])
        tower7 = self.tower7(bert_outputs[6])
 
        info1 = torch.unsqueeze(self.info1(tower1), 1) # self.info1(tower1): [7, 43, 32] [batch_size, seq_len, 32]
        # print(self.info1(tower1).shape, info1.shape)
        info2 = torch.unsqueeze(self.info2(tower2), 1) # info1: [7, 1, 43, 32] [batch_size, 1, seq_len, 32]
        info3 = torch.unsqueeze(self.info3(tower3), 1)
        info4 = torch.unsqueeze(self.info4(tower4), 1)
        info5 = torch.unsqueeze(self.info5(tower5), 1)
        info6 = torch.unsqueeze(self.info6(tower6), 1)
        info7 = torch.unsqueeze(self.info7(tower7), 1)

        ait1 = self.attention1(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1)) # torch.cat([info1, info2, info3, info4, info5, info6, info7], 1): [7, 7, 43, 32] [batch_size, batch_size, seq_len, 32]
        # print(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1).shape, ait1.shape)
        ait2 = self.attention2(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1)) # ait1: [7, 43, 32] [batch_size, seq_len, 32]
        ait3 = self.attention3(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1))
        ait4 = self.attention4(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1))
        ait5 = self.attention5(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1))
        ait6 = self.attention6(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1))
        ait7 = self.attention7(torch.cat([info1, info2, info3, info4, info5, info6, info7], 1))

        start_logits.append(self.start_outputs1(ait1).squeeze(-1)) # self.start_outputs1(ait1): [7, 43, 1] [batch_size, seq_len, 1]
        # print(self.start_outputs1(ait1).shape, start_logits[0].shape)
        end_logits.append(self.end_outputs1(ait1).squeeze(-1)) # start_logits: [..., [batch_size, seq_len]]
        # print(ait1.unsqueeze(2).shape, ait1.unsqueeze(1).shape)
        start_extend = ait1.unsqueeze(2).expand(-1, -1, seq_len, -1) # ait1.unsqueeze(2): [7， 43， 1, 32]
        end_extend = ait1.unsqueeze(1).expand(-1, seq_len, -1, -1) # ait1.unsqueeze(1): [7, 1, 43, 32]
        # print(start_extend.shape, end_extend.shape)
        span_matrix = torch.cat([start_extend, end_extend], 3) # start_extend: [7, 43, 43, 32]
        span_logits.append(self.span_embedding1(span_matrix).squeeze(-1))
        
        start_logits.append(self.start_outputs2(ait2).squeeze(-1))
        end_logits.append(self.end_outputs2(ait2).squeeze(-1))
        start_extend = ait2.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait2.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding2(span_matrix).squeeze(-1))
        
        start_logits.append(self.start_outputs3(ait3).squeeze(-1))
        end_logits.append(self.end_outputs3(ait3).squeeze(-1))
        start_extend = ait3.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait3.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding3(span_matrix).squeeze(-1))
        
        start_logits.append(self.start_outputs4(ait4).squeeze(-1))
        end_logits.append(self.end_outputs4(ait4).squeeze(-1))
        start_extend = ait4.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait4.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding4(span_matrix).squeeze(-1))
        
        start_logits.append(self.start_outputs5(ait5).squeeze(-1))
        end_logits.append(self.end_outputs5(ait5).squeeze(-1))
        start_extend = ait5.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait5.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding5(span_matrix).squeeze(-1))
        
        start_logits.append(self.start_outputs6(ait6).squeeze(-1))
        end_logits.append(self.end_outputs6(ait6).squeeze(-1))
        start_extend = ait6.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait6.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding6(span_matrix).squeeze(-1))
        
        start_logits.append(self.start_outputs7(ait7).squeeze(-1))
        end_logits.append(self.end_outputs7(ait7).squeeze(-1))
        start_extend = ait7.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = ait7.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits.append(self.span_embedding7(span_matrix).squeeze(-1))

        new_start_logits = []
        new_end_logits = []
        new_span_logits = []
        for i in range(int(len(input_ids)/7)):
            for j in range(7):
                new_start_logits.append(start_logits[j][i])
                new_end_logits.append(end_logits[j][i])
                new_span_logits.append(span_logits[j][i])
        new_start_logits = torch.stack(new_start_logits)
        new_end_logits = torch.stack(new_end_logits)
        new_span_logits = torch.stack(new_span_logits)
        # print(new_start_logits.shape)

        return new_start_logits, new_end_logits, new_span_logits

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            elif self.span_loss_candidates == "pred_gold_random":
                gold_and_pred = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
                data_generator = torch.Generator()
                data_generator.manual_seed(0)
                random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
                random_matrix = random_matrix.cuda()
                match_candidates = torch.logical_or(
                    gold_and_pred, random_matrix
                )
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss