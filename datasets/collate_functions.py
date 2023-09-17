#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: collate_functions.py

import torch
from typing import List


def tagger_collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, attention_mask, wordpiece_label_idx_lst
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(3):
        # 0 -> tokens
        # 1 -> token_type_ids
        # 2 -> attention_mask
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # 3 -> sequence_label
    # -100 is ignore_index in the cross-entropy loss function.
    pad_output = torch.full([batch_size, max_length], -100, dtype=batch[0][3].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    # 4 -> is word_piece_label
    pad_output = torch.full([batch_size, max_length], -100, dtype=batch[0][4].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][4]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    return output


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.full([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))

    return output

def multi_collate_to_max_length(batch: List[List[List[torch.Tensor]]]) -> List[List[torch.Tensor]]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains #num_class list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, num_class, max_length]
    """
    batch_size = len(batch)
    num_class = len(batch[0])
    # print("\nbatch_size: ", batch_size, "num_class: ", num_class)

    max_length = 0
    for i in range(batch_size):
        max_length = max(max_length, max(x[0].shape[0] for x in batch[i]))

    output = []
    for field_idx in range(6):
        # [batch_size, max_length]
        pad_output = torch.full([batch_size, num_class, max_length], 0, dtype=batch[0][0][field_idx].dtype)
        for sample_idx in range(batch_size):
            for class_idx in range(num_class):
                data = batch[sample_idx][class_idx][field_idx]
                pad_output[sample_idx][class_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, num_class, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        for class_idx in range(num_class):
            data = batch[sample_idx][class_idx][6]
            pad_match_labels[sample_idx, class_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    for field_idx in range(7,9):
        # [batch_size, max_length]
        pad_output = torch.full([batch_size, num_class, max_length], 0, dtype=batch[0][0][field_idx].dtype)
        for sample_idx in range(batch_size):
            for class_idx in range(num_class):
                data = batch[sample_idx][class_idx][field_idx]
                pad_output[sample_idx][class_idx][: data.shape[0]] = data
        output.append(pad_output)

    return output


# def multi_encoder_collate_to_max_length(batch: List[List[List[torch.Tensor]]]) -> List[List[torch.Tensor]]:
#     """
#     pad to maximum length of this batch
#     Args:
#         batch: a batch of samples, each contains #num_class list of field data(Tensor):
#             tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
#     Returns:
#         output: list of field batched data, which shape is [batch, num_class, max_length]
#     """
#     batch_size = len(batch)
#     num_class = len(batch[0])
#     # print("\nbatch_size: ", batch_size, "num_class: ", num_class)

#     # max_length = 0
#     # for i in range(batch_size):
#     #     for j in range(num_class):
#     #         max_length = max(max_length, max(x.shape[0] for x in batch[i][j]))
#     #         print("batch[i][-1].shape", batch[i][j][-1].shape)
#     #         print(batch[i][j][0].shape, batch[i][j][1].shape, batch[i][j][2].shape, batch[i][j][3].shape, max_length)
#     #         max_length = max(max_length, batch[i][j][1].shape[1])
#     max_length = 0
#     for i in range(batch_size):
#         # print(len(batch[i]), len(batch[i][0]), batch[i][0][0].shape)
#         max_length = max(max_length, max(x[2].shape[0] for x in batch[i]))
#         # print("======\n", len(batch[i][0]), batch[i][0][1].shape)
#         # max_length = max(max_length, batch[i][1].shape[1])

#     output = []
#     for field_idx in range(2):
#         pad_output = torch.full([batch_size, num_class, max_length, 768], 0, dtype=batch[0][0][field_idx].dtype)
#         for sample_idx in range(batch_size):
#             for class_idx in range(num_class):
#                 data = batch[sample_idx][class_idx][field_idx]
#                 # print(data.shape, type(data))
#                 data = data.repeat(max_length, 1)
#                 # print(data.shape)
#                 pad_output[sample_idx][class_idx][: data.shape[0]] = data
#         output.append(pad_output)


#     for field_idx in range(2, 6):
#         # [batch_size, max_length]
#         pad_output = torch.full([batch_size, num_class, max_length], 0, dtype=batch[0][0][field_idx].dtype)
#         for sample_idx in range(batch_size):
#             for class_idx in range(num_class):
#                 data = batch[sample_idx][class_idx][field_idx]
#                 pad_output[sample_idx][class_idx][: data.shape[0]] = data
#         output.append(pad_output)

#     pad_match_labels = torch.zeros([batch_size, num_class, max_length, max_length], dtype=torch.long)
#     for sample_idx in range(batch_size):
#         for class_idx in range(num_class):
#             data = batch[sample_idx][class_idx][6]
#             pad_match_labels[sample_idx, class_idx, : data.shape[1], : data.shape[1]] = data
#     output.append(pad_match_labels)

#     return output


def multi_encoder_collate_to_max_length(batch: List[List[List[torch.Tensor]]]) -> List[List[torch.Tensor]]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains #num_class list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, num_class, max_length]
    """
    batch_size = len(batch)
    num_class = len(batch[0])
    # print("\nbatch_size: ", batch_size, "num_class: ", num_class)

    max_length = 0
    for i in range(batch_size):
        for j in range(num_class):
            max_length = max(max_length, max(x.shape[0] for x in batch[i][j]))
            # print("batch[i][j][-1].shape", batch[i][j][-1].shape)
            # print(batch[i][j][-1].shape[1], max_length)
            max_length = max(max_length, batch[i][j][-1].shape[1])

    output = []
    for field_idx in range(6):
        # [batch_size, max_length]
        pad_output = torch.full([batch_size, num_class, max_length], 0, dtype=batch[0][0][field_idx].dtype)
        for sample_idx in range(batch_size):
            for class_idx in range(num_class):
                data = batch[sample_idx][class_idx][field_idx]
                pad_output[sample_idx][class_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, num_class, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        for class_idx in range(num_class):
            data = batch[sample_idx][class_idx][6]
            pad_match_labels[sample_idx, class_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    for field_idx in range(7,9):
        # [batch_size, max_length]
        pad_output = torch.full([batch_size, num_class, max_length], 0, dtype=batch[0][0][field_idx].dtype)
        for sample_idx in range(batch_size):
            for class_idx in range(num_class):
                data = batch[sample_idx][class_idx][field_idx]
                pad_output[sample_idx][class_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_output = torch.full([batch_size, num_class, max_length, 50], 0, dtype=batch[0][0][9].dtype)
    # print("char", batch[0][0][-1].shape, batch[0][0][0].shape)
    for sample_idx in range(batch_size):
        for class_idx in range(num_class):
            data = batch[sample_idx][class_idx][-1]
            # print("\n\n++++++++\n")
            # print("data.shape", max_length, data.shape, pad_output[sample_idx][class_idx][: data.shape[1]].shape)
            pad_output[sample_idx][class_idx][: data.shape[1]] = data.squeeze(0)
    output.append(pad_output)

    return output

    # batch_size = len(batch)
    # max_length = max(x[1].shape[0] for x in batch)
    # output = []

    # pad_output_token = []
    # for sample_idx in range(batch_size):
    #     data = batch[sample_idx][0]
    #     pad_output_token.append(data)
    # output.append(pad_output_token) 

    # for field_idx in range(1,7):
    #     pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
    #     for sample_idx in range(batch_size):
    #         data = batch[sample_idx][field_idx]
    #         pad_output[sample_idx][: data.shape[0]] = data
    #     output.append(pad_output)

    # pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    # for sample_idx in range(batch_size):
    #     data = batch[sample_idx][7]
    #     pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    # output.append(pad_match_labels)

    # output.append(torch.stack([x[-2] for x in batch]))
    # output.append(torch.stack([x[-1] for x in batch]))

    # return output
