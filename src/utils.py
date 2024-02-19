import warnings
from tqdm import tqdm
import os
import pickle
warnings.filterwarnings(action='ignore')
import guidance

from retriever import Index_Builder
from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoTokenizer
import openai
from secret import api_key
import faiss
import numpy as np
from secret import api_key


def generate_answer(model_name, api_key, prompt, **kwargs):
    
    llm = guidance.llms.OpenAI(
                        model_name,#"gpt-3.5-turbo",#"gpt-4",#"gpt-3.5-turbo", 
                        caching=False,
                        api_key = api_key
                        ) 

    guidance.llm = llm 
    
    guidance.llm.cache.clear()

    structure_program = guidance(prompt, silent = True,**kwargs)
                            
    res = structure_program()

    return res

def Retrieve_documents(
                    query, data_dir,
                    embedding_data_dir,
                    use_content_type = 'All',
                    top_k: int = 20, 
                    index_save_dir = None,
                    retrieve_mode = 'sparse'
                    ):
    
    
    if retrieve_mode == 'dense':
        
        builder = Index_Builder(
                        sparse_tokenizer = None,
                        data_dir = data_dir,
                        index_save_dir = index_save_dir,
                        use_content_type = 'All',
                        batch_size = 64,
                        retrieve_mode = retrieve_mode
                        )


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