import warnings
from tqdm import tqdm
import os
import pickle
warnings.filterwarnings(action='ignore')
import guidance

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

def generate_answer(model_name, api_key, prompt, **kwargs):
    
    llm = guidance.llms.OpenAI(
                        "gpt-4",#"gpt-3.5-turbo", 
                        caching=False,
                        api_key = "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"#api_key #"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",
                        ) 

    guidance.llm = llm 
    
    guidance.llm.cache.clear()

    structure_program = guidance(prompt, silent = True,**kwargs)
                            
    res = structure_program()

    return res

def Retrieve_documents(
                    query, data_dir,
                    sparse_tokenizer,
                    use_content_type = 'All',
                    top_k: int = 20, 
                    check_point_dir = None,
                    index_save_dir = None,
                    retrieve_mode = 'sparse'
                    ):
    
    if retrieve_mode == 'dense':

        
        tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        model = KobertBiEncoder()
        model.cuda()
        model.load(check_point_dir)
        model.eval()
        
        builder = Index_Builder(
                        model = model, 
                        tokenizer=tokenizer,
                        data_dir = data_dir,
                        index_save_dir = index_save_dir,
                        use_content_type = 'All',
                        batch_size = 64,
                        retrieve_mode = retrieve_mode
                        )
        
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