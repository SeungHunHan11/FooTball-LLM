import re
import os
import json
import subprocess
import warnings
import pickle
import torch
import faiss
import argparse
import numpy as np
from tqdm import tqdm

from adaptertransformers.src import transformers
from adaptertransformers.src.transformers import AutoConfig, AutoTokenizer
from adaptertransformers.src.transformers import PretrainedConfig
from dense_model import BertDense
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import glob

def load_json_data(data_dir):
    """
    Load multiple JSON files from the folder and merge.
    """

    files = glob.glob(data_dir + "/*.json")
    files.sort()
    all_data = []
    for file in files:
        print("Loading: ",file)
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r", encoding = "utf-8-sig") as f:
            doc = json.load(f)
        all_data.append(doc)
        #all_data += doc
    return all_data

def build_model(self, dam_path):
    """
    Building model for converting document contents into embeddings,
        the model includes DAM module and REM module.
    """

    config = PretrainedConfig.from_pretrained(dam_path)
    config.similarity_metric, config.pooling = "ip", "average"
    tokenizer = AutoTokenizer.from_pretrained(dam_path, config=config)
    model = BertDense.from_pretrained(dam_path, config=config)
    adapter_name = model.load_adapter(self.rem_url)
    model.set_active_adapters(adapter_name)
    return model, tokenizer

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tokenized_text["input_ids"])
        attention_mask = torch.tensor(tokenized_text["attention_mask"])
        return input_ids, attention_mask



def train_dam(index_save_dir, use_content_type, all_docs, batch_size):
    """
    Training a DAM module based on unsupervised methods.
    
    Firstly, it is necessary to convert contents data in JSON files into TSV format, 
            and then use the script provided by the disentangled_retriever library for training.
    Reference: https://github.com/jingtaozhan/disentangled-retriever/blob/main/examples/domain_adapt/chinese-dureader/adapt_to_new_domain.md
    """

    # build corpus for training
    corpus_path = index_save_dir + '/corpus.tsv'
    with open(corpus_path,"w") as f:
        for idx, doc in enumerate(all_docs):
            # remove special character
            if use_content_type == "title":
                doc_content = doc['title']
            elif use_content_type == "contents":
                try:
                    doc_content = doc['contents']
                except:
                    doc_content = doc['content']

            else:
                try:
                    doc_content = doc['title'] + doc['contents']
                except:
                    doc_content = doc['title'] + doc['content']
                    
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '',doc_content)
            f.write("{}\t{}\n".format(idx,content))

    # train DAM module
    dam_path = index_save_dir + '/dam_module'

    init_model = "jingtao/DAM-bert_base-mlm-msmarco"
    # set batch size for training
    batch_size = batch_size #64 if len(all_docs) > 64 else len(all_docs)
    training_args = ["--nproc_per_node", "1",
                        "-m", "train_dam_module",
                        "--corpus_path", corpus_path,
                        "--output_dir", dam_path,
                        "--model_name_or_path", init_model,
                        "--max_seq_length", "512",
                        "--gradient_accumulation_steps", "1",
                        "--per_device_train_batch_size", str(batch_size),
                        "--warmup_steps", "1000",
                        "--fp16",
                        "--learning_rate", "2e-5",
                        "--max_steps", "400000",
                        "--dataloader_drop_last",
                        "--overwrite_output_dir",
                        "--weight_decay", "0.01",
                        "--save_steps", "5000",
                        "--lr_scheduler_type", "constant_with_warmup",
                        "--save_strategy", "steps",
                        "--optim", "adamw_torch"]
    subprocess.run(["torchrun"] + training_args)
    return dam_path

def main():
    dam_path = "jingtao/DAM-bert_base-mlm-msmarco"

    rem_url = "https://huggingface.co/jingtao/REM-bert_base-dense-distil-msmarco/resolve/main/lora192-pa4.zip"
    

    index_save_dir = '/Project/index_save_dir'
    use_content_type = 'all'
    all_docs = load_json_data('/Project/json_data_new')
    dense_index_path = index_save_dir + "/dense"
    #faiss_save_path = dense_index_path + "/faiss.index"
    
    train_dam(index_save_dir, use_content_type, all_docs, batch_size=8)
    
if __name__=="__main__":
    main()