import torch

import torch
import torch.nn as nn
from copy import deepcopy
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi

from torch.utils.data import DataLoader
import faiss

import logging
import os
import json
from tqdm import tqdm
from glob import glob
import pickle

from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab

import numpy as np
import pandas as pd

class KobertBiEncoder(torch.nn.Module):
    def __init__(self):
        super(KobertBiEncoder, self).__init__()
        
        self.passage_encoder = AutoModel.from_pretrained("skt/kobert-base-v1")     
        self.query_encoder = AutoModel.from_pretrained("skt/kobert-base-v1")
            
    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, type: str
    ) -> torch.FloatTensor:
        """passage 또는 query를 bert로 encoding합니다."""

        if type == "passage":
            return self.passage_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output
        else:
            return self.query_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output

    def checkpoint(self, model_ckpt_path):
        torch.save(deepcopy(self.state_dict()), model_ckpt_path)

    def load(self, model_ckpt_path):
        with open(model_ckpt_path, "rb") as f:
            state_dict = torch.load(f, map_location='cuda:0')
        self.load_state_dict(state_dict)    
    

class Index_Builder:
    
    def __init__(self, 
                 sparse_tokenizer,
                 data_dir, 
                 index_save_dir, use_content_type = 'All', 
                 batch_size = 32, retrieve_mode = 'sparse'):
        
        assert retrieve_mode in ['dense', 'sparse'], "retrieve_mode should be either 'dense' or 'sparse'"
        
        #os.makedirs(index_save_dir, exist_ok = True)

        self.retrieve_mode = retrieve_mode
        self.sparse_tokenizer = sparse_tokenizer
        self.data_dir = data_dir
        self.index_save_dir = index_save_dir
        self.use_content_type = use_content_type
        self.all_docs = load_json_data(self.data_dir)
        
        if self.use_content_type == "title":
            self.doc_content = [item['title'] for item in self.all_docs]
        elif self.use_content_type == "content":
            self.doc_content = [item['contents'] for item in self.all_docs]
        else:
            self.doc_content = []
            for item in tqdm(self.all_docs):
                try:
                    self.doc_content.append(item['title'] + item['contents'])
                except:
                    pass
                    #self.doc_content.append(item['title'] + item['content'])
        
        if retrieve_mode == 'sparse':
            # corpus = []
            # doc_id_ref = []
            # #tokenizer = Mecab()

            # for idx, passage in tqdm(enumerate(self.doc_content)):

            #     tokenized_doc = sparse_tokenizer.convert_ids_to_tokens(sparse_tokenizer(passage,add_special_tokens = False )['input_ids'])
            #     #tokenized_doc = tokenizer.morphs(passage)
            #     corpus.append(tokenized_doc)
            #     doc_id_ref.append(idx)
                
            # self.corpus = corpus
            # self.doc_id_ref = doc_id_ref
            # load corpus pickle
            with open('/Project/src/index_save_dir/sparse/corpus.pkl', 'rb') as f:
                self.corpus = pickle.load(f)
                
            self.doc_id_ref = list(range(len(self.corpus))) 
            
            
    def load_data(self, batch_size):
        
        # load json files
        #print("Start loading data...")
        #print("Finish.")
        
        # convert doc to vector
        #print("Start converting documents to vectors...")
        
        doc_dataset = TextDataset(self.doc_content, self.tokenizer)
        self.doc_loader = DataLoader(doc_dataset, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn,
                                shuffle = False
                                )
        
        return self.doc_loader
        
    def build_dense_index(self, data_dir):
        """
        Building dense retrieval index based on faiss.

        Firstly, use model to conver contents in json files to embeddings. 
        Then, build faiss index based on embeddings.
        """

        os.makedirs(self.index_save_dir, exist_ok = True)
        faiss_save_path = os.path.join(self.index_save_dir, "ada_embedding.faiss")
        
        #data_dir = '/Project/embedding_data'
        files = glob(os.path.join(data_dir,'*.json'))
        files.sort()

        title = []
        content = []
        embedding = []

        for file_path in files:

            with open(file_path, "r", encoding = "utf-8-sig") as f:
                doc = json.load(f)
            title.append(doc['title'])
            content.append(doc['contents'])
            embedding.append(doc['embedding'])
    
        df = pd.DataFrame({'title':title, 'content':content, 'ada_embedding':embedding})
        
        embeddings = df['ada_embedding'].apply(np.array).to_numpy()
        embeddings = np.float32(np.stack(embeddings))
        
        # Build faiss index by using doc embedding
        print("Start building faiss index...")

        index = faiss.IndexFlatL2(embeddings.shape[1])  # Example of a simple flat L2 index
        index.add(embeddings)  # Add some vectors to the index
        faiss.write_index(index, faiss_save_path) #'/Project/ada_embeddings.faiss')
        print("Finish building index.")
        
        self.dense_index = index
        
        return df, index
        
    def build_sparse_index(self):

        self.searcher = BM25Okapi(self.corpus)
        
        #Save the BM25 index to a file
        with open(os.path.join(self.index_save_dir,'bm25_index.pickle'), 'wb') as index_file:
            pickle.dump(self.searcher, index_file)

        return self.searcher

        
    def search(self, query, top_k: int = 20, searcher = None):
        raw_reference_list = []

        if self.retrieve_mode == 'dense':
            
            files = glob(os.path.join('../embedding_data','*.json'))
            files.sort()
            data_dir = '../embedding_data_wiki'
            files_wiki = glob(data_dir+"/*.json")
            files_wiki.sort()

            #files = files + files_wiki
            files.sort()

            index = faiss.read_index("/Project/src/index_save_dir/dense/ada_embeddings.faiss")#os.path.join(self.index_save_dir,'ada_embedding.faiss')) #"/Project/ada_embeddings.faiss")
            faiss.normalize_L2(query)
            
            D, idx = index.search(query, top_k)
        
            for i,id in enumerate(idx[0]):
                with open(files[id], "r", encoding = "utf-8-sig") as f:
                    raw_reference = json.load(f)
                # try:
                #     raw_reference = "제목: " + raw_reference['title'] + "본문:" + raw_reference['contents']
                # except:
                #     raw_reference = "제목: " + raw_reference['title'] + "본문:" + raw_reference['content']
                
                raw_reference_list.append(raw_reference)
                
        else:            
            self.query = self.sparse_tokenizer.convert_ids_to_tokens(self.sparse_tokenizer(query,add_special_tokens = False)['input_ids'])
            #tokenizer = Mecab()
                        
            if searcher is not None:
                self.searcher = searcher
            else:
                self.searcher = self.build_sparse_index()
                
            print('Searching Sparse Index...')
            scores = self.searcher.get_scores(self.query)

            idx = np.argsort(scores)[::-1]
            top_k_indices = idx[:top_k]

            for idx in top_k_indices:

                raw_reference = self.all_docs[idx]
                raw_reference_list.append(raw_reference)

        raw_contents = []

        for item in raw_reference_list:
            try:
                raw_contents.append(item['title'] + item['contents'])
            except:
                raw_contents.append(item['title'] + item['content'])
            
        return raw_reference_list, raw_contents
    
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
    
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return input_ids, attention_masks

def load_json_data(data_dir):
    """
    Load multiple JSON files from the folder and merge.
    """

    files = glob(data_dir+"/*.json")
    files.sort()
    all_data = []
    for file_path in files:
        #print("Loading: ",file)
        #file_path = os.path.join(data_dir, file)
        with open(file_path, "r", encoding = "utf-8-sig") as f:
            doc = json.load(f)
        all_data.append(doc)
        #all_data += doc
    return all_data

