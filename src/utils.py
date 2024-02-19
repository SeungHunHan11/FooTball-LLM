import warnings
from tqdm import tqdm
import os
import pickle
warnings.filterwarnings(action='ignore')
import guidance

<<<<<<< HEAD
from retriever import Index_Builder
from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoTokenizer
import openai
from secret import api_key
import faiss
import numpy as np
from secret import api_key

=======
from retriever import KobertBiEncoder, Index_Builder
from kobert_tokenizer import KoBERTTokenizer

import faiss

from secret import api_key

def load_model(model_name, api_key):
    
    llm = guidance.llms.OpenAI(
                            model_name, #"gpt-3.5-turbo", 
                            caching=True,
                            api_key = api_key #"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",
                            ) 
    
    guidance.llm = llm 
    guidance.llm.cache.clear()
    
    return guidance
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b

def generate_answer(model_name, api_key, prompt, **kwargs):
    
    llm = guidance.llms.OpenAI(
<<<<<<< HEAD
                        model_name,#"gpt-3.5-turbo",#"gpt-4",#"gpt-3.5-turbo", 
                        caching=False,
                        api_key = api_key
=======
                        "gpt-4",#"gpt-3.5-turbo", 
                        caching=False,
                        api_key = "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"#api_key #"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",
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b
                        ) 

    guidance.llm = llm 
    
    guidance.llm.cache.clear()

    structure_program = guidance(prompt, silent = True,**kwargs)
                            
    res = structure_program()

    return res

def Retrieve_documents(
                    query, data_dir,
<<<<<<< HEAD
                    embedding_data_dir,
                    use_content_type = 'All',
                    top_k: int = 20, 
=======
                    sparse_tokenizer,
                    use_content_type = 'All',
                    top_k: int = 20, 
                    check_point_dir = None,
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b
                    index_save_dir = None,
                    retrieve_mode = 'sparse'
                    ):
    
<<<<<<< HEAD
    
    if retrieve_mode == 'dense':
        
        builder = Index_Builder(
                        sparse_tokenizer = None,
=======
    if retrieve_mode == 'dense':

        
        tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        model = KobertBiEncoder()
        model.cuda()
        model.load(check_point_dir)
        model.eval()
        
        builder = Index_Builder(
                        model = model, 
                        tokenizer=tokenizer,
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b
                        data_dir = data_dir,
                        index_save_dir = index_save_dir,
                        use_content_type = 'All',
                        batch_size = 64,
                        retrieve_mode = retrieve_mode
                        )
<<<<<<< HEAD


        if not index_save_dir:
            df, dense_index = builder.build_dense_index(embedding_data_dir)
        
        openai.api_key = api_key
        query_output = openai.Embedding.create(input = query, model = 'text-embedding-ada-002')
        query_embedding = np.float32(query_output['data'][0]['embedding']).reshape(1,-1)
        
        raw_reference_list  = builder.search(query = query_embedding,
                                                top_k = top_k,
                                                )
        
    else:
        sparse_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

        builder = Index_Builder(
=======
        
        dense_index = builder.build_dense_index()
        searcher = dense_index

        if not index_save_dir:
            dense_index = builder.build_dense_index(batch_size=64)
        else:
            dense_index = faiss.read_index(index_save_dir)
            
        raw_reference_list  = builder.search(query = query,
                                                top_k = top_k,
                                                searcher = dense_index
                                                )
    else:
        
        builder = Index_Builder(model = None,
                tokenizer = None,
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b
                sparse_tokenizer = sparse_tokenizer,
                data_dir = data_dir,
                index_save_dir = index_save_dir,
                use_content_type = 'All',
                batch_size = 64,
                retrieve_mode = retrieve_mode
                )
            
        saved_index_path = os.path.join(index_save_dir,'bm25_index.pickle')

        print(saved_index_path)
        if os.path.exists(saved_index_path):
            # Load the BM25 index from a file
            with open(saved_index_path, 'rb') as index_file:
                loaded_bm25 = pickle.load(index_file)
            print("Loading Saved sparse index")
        else:
            print('Building sparse index')
            loaded_bm25 = builder.build_sparse_index()
        
        raw_reference_list = builder.search(query = query, 
                                                top_k = top_k,
                                                searcher = loaded_bm25)
        
    return raw_reference_list