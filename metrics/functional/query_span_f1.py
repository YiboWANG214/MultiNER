#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: query_span_f1.py

import torch
import numpy as np
from utils.bmes_decode import bmes_decode
import json

class_dict = {
    0: "GPE",
    1: "ORG",
    2: "PER",
    3: "FAC",
    4: "VEH",
    5: "LOC",
    6: "WEA"
}

def query_exact_noempty_f1(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, match_labels, flat=False):
    # print("match_preds", match_preds.shape)
    # print("match_labels", match_labels.shape)
    # print("start_label_mask", start_label_mask.shape)
    # print(start_label_mask)

    match_labels = match_labels.bool()
    # print(match_preds)
    match_preds = match_preds > 0
    # print(match_preds)

    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()
    bsz, seq_len = start_label_mask.size()
    # print("start_preds", start_preds.shape, start_preds.unsqueeze(-1).shape) # [batch_size, num_class, seq], [batch_size, num_class, seq, 1]
    # print(match_preds.shape)
    # print(end_preds.unsqueeze(-2).shape)
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    # print("\n\n", start_label_mask.shape)
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    match_preds = match_preds.bool()

    label_count = match_labels.sum()-match_labels.sum()
    pred_count = match_labels.sum()-match_labels.sum()
    correct = match_labels.sum()-match_labels.sum()
    # print("label_count, pred_count, correct", label_count, pred_count, correct)
    for i in range(len(match_labels)):
        for j in range(len(match_labels[i])):
            if torch.any(match_labels[i][j]):
                for k in range(len(match_labels[i][j])):
                    if match_label_mask[i][j][k] and match_labels[i][j][k]:
                        label_count += match_labels[i][j][k].sum()
                        if torch.equal(match_labels[i][j][k], match_preds[i][j][k]):
                            correct += match_labels[i][j][k].sum()
            if torch.any(match_preds[i][j]):
                for k in range(len(match_preds[i][j])):
                    if match_label_mask[i][j][k] and match_preds[i][j][k]:
                        pred_count += match_preds[i][j][k].sum()

    return torch.stack([label_count, pred_count, correct])


def query_span_noempty_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    # print("start_preds", start_preds.shape, start_preds.unsqueeze(-1).shape) # [batch_size, num_class, seq], [batch_size, num_class, seq, 1]
    # print(match_preds.shape)
    # print(end_preds.unsqueeze(-2).shape)
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    # print("\n\n", start_label_mask.shape)
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    new_match_labels = []
    new_match_preds = []
    num, seq_len1, seq_len2 = match_labels.shape
    for i in range(num):
        if torch.any(match_labels[i]):
            new_match_labels.append(match_labels[i])
            new_match_preds.append(match_preds[i])
    if new_match_labels:
        new_match_labels = torch.stack(new_match_labels)
        new_match_preds = torch.stack(new_match_preds)
        tp = (new_match_labels & new_match_preds).long().sum()
        fp = (~new_match_labels & new_match_preds).long().sum()
        fn = (new_match_labels & ~new_match_preds).long().sum()
    else:
        tp = torch.tensor(0)
        fp = torch.tensor(0)
        fn = torch.tensor(0)


    # print("torch.stack([tp, fp, fn])", torch.stack([tp, fp, fn]).shape, match_labels.shape)
    return torch.stack([tp, fp, fn])


def query_type_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    # print("\n\n", start_label_mask.shape)
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    num, seq_len1, seq_len2 = match_labels.shape

    stat = []
    for i in range(num):
        tp = (match_labels[i] & match_preds[i]).long().sum()
        fp = (~match_labels[i] & match_preds[i]).long().sum()
        fn = (match_labels[i] & ~match_preds[i]).long().sum()
        stat.append(torch.stack([tp, fp, fn]))
        # print("=========\n\n")
    for i in range(num):
        if torch.any(match_labels[i]):
            type_count = torch.tensor(1).to('cuda:1')
        else:
            type_count = torch.tensor(0).to('cuda:1')
        stat.append(torch.stack([type_count, type_count, type_count]))
    # print(count)
    return torch.stack(stat)


def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    # print("start_preds", start_preds.shape, start_preds.unsqueeze(-1).shape) # [batch_size, num_class, seq], [batch_size, num_class, seq, 1]
    # print(match_preds.shape)
    # print(end_preds.unsqueeze(-2).shape)
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    # print("\n\n", start_label_mask.shape)
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    print(match_preds.shape)
    for k in range(7):
        for i in range(len(match_preds[k])):
            for j in range(len(match_preds[k][i])):
                if match_preds[k][i][j]:
                    print("preds", i, j)
                    print('\n')
                
                if match_labels[k][i][j]:
                    print("labels", i, j)
                    print('\n')
            
        print("===============")

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])


def extract_nested_spans(start_preds, end_preds, match_preds, start_label_mask, end_label_mask, pseudo_tag="TAG"):
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    bsz, seq_len = start_label_mask.size()
    start_preds = start_preds.bool()
    end_preds = end_preds.bool()

    match_preds = (match_preds & start_preds.unsqueeze(-1).expand(-1, -1, seq_len) & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len) & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    match_pos_pairs = np.transpose(np.nonzero(match_preds.numpy())).tolist()
    return [(pos[0], pos[1], pseudo_tag) for pos in match_pos_pairs]


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, pseudo_tag = "TAG"):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(entity.begin, entity.end, entity.tag) for entity in tags]


def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    output = []
    occupied = set()
    for start, end in spans:
        if any(x for x in range(start, end+1)) in occupied:
            continue
        output.append((start, end))
        for x in range(start, end + 1):
            occupied.add(x)
    return output
