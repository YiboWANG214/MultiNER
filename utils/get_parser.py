#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: get_parser.py

import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--max_keep_ckpt", default=3, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--bert_config_dir", type=str, required=True, help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", default=0, type=int, help="set random seed for reproducing results.")
    parser.add_argument("--data_name", type=str, help="dataset name")

    # # new
    # parser.add_argument("--bert_before_lstm", action="store_true")
    # parser.add_argument("--subword_aggr", type=str, default="first",
    #                     choices=['first', 'mean', 'max'])
    # parser.add_argument("--bert_output", type=str, default="last",
    #                     choices=['last', 'concat-last-4', 'mean-last-4'])
    # parser.add_argument("--reinit", type=int, default=0)
    
    # parser.add_argument("--word", action="store_true")
    # parser.add_argument("--word_embed", type=str, default="")
    # parser.add_argument("--word_dp", type=float, default=0.5)
    # parser.add_argument("--word_freeze", action="store_true")
    
    # parser.add_argument("--char", action="store_true")
    # parser.add_argument("--char_layer", type=int, default=1)
    # parser.add_argument("--char_dim", type=int, default=50)
    # parser.add_argument("--char_dp", type=float, default=0.2)
    
    # parser.add_argument("--pos", action="store_true")
    # parser.add_argument("--pos_dim", type=int, default=50)
    # parser.add_argument("--pos_dp", type=float, default=0.2)
    
    # parser.add_argument("--agg_layer", type=str, default="lstm",
    #                     choices=["lstm", "transformer"])
    # parser.add_argument("--lstm_dim", type=int, default=1024)
    # parser.add_argument("--lstm_layer", type=int, default=1)
    # parser.add_argument("--lstm_dp", type=float, default=0.2)
    
    # parser.add_argument("--tag", type=str, default="")
    return parser
