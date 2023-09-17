import os
import torch
from tokenizers import BertWordPieceTokenizer, Tokenizer

from torch.utils.data import DataLoader

from datasets.mm_ner_dataset import MMNERDataset
from utils.random_seed import set_random_seed
import pickle

set_random_seed(0)

def get_dataset(prefix="train", limit: int = None) -> DataLoader:
    """get training dataloader"""
    """
    load_mmap_dataset
    """
    json_path = os.path.join(DATA_DIR, f"mrc-ner.{prefix}")
    print(json_path)
    vocab_path = os.path.join("/mnt/WDRed4T/yibo/multi_task_NER/mrc-for-flat-nested-ner/BERT/bert-base-uncased", "bert-base-uncased-vocab.txt")
    dataset = MMNERDataset(json_path=json_path,
                            tokenizer=BertWordPieceTokenizer(vocab_path),
                            # tokenizer=BertTokenizer.from_pretrained(self.bert_dir),
                            # tokenizer=Tokenizer.from_pretrained(self.bert_dir),
                            max_length=MAX_LENGTH,
                            is_chinese=False,
                            pad_to_maxlen=False
                            )

    return dataset


DATA_DIR = "/mnt/WDRed4T/yibo/multi_task_NER/proposed_model/data/ace2004"
MAX_LENGTH = 128

for tag in ['train', 'dev', 'test']:
    dataset = get_dataset(tag)
    with open(os.path.join(DATA_DIR, f"embedding.{tag}"), 'wb') as data_file:
        pickle.dump(dataset, data_file)
