from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
from typing import List, Optional, Tuple, Union
import torch
import json
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import streamlit as st
import openai
import os
import faiss
import warnings
from tqdm import tqdm
#from peft import PeftConfig, PeftModel
warnings.filterwarnings(action='ignore')

import pickle
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from kobert_tokenizer import KoBERTTokenizer

from config.prompt_templates import global_no_demon_template, global_template, request_rewriting_template, passage_extraction_template, answer_generation_template, fact_checking_template

from model.retriever import KobertBiEncoder, Index_Builder

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"


class load_models():
    
    def __init__(self, model_config_path, device):

        self.model_config_path = model_config_path
        self.device = device
        self.model_config = json.load(open(model_config_path, "r"))
        self.model = self.get_llm()

    def get_llm(self):
            
        if ("chatgpt" in self.model_config_path):
            model = "chatgpt"
            #tokenizer = ""
            
        if ("alpaca" in self.model_config_path) :
            model_config = json.load(open(self.model_config_path, "r"))

            MODEL = model_config['model_path']
            device = self.device
            
            #config = PeftConfig.from_pretrained(MODEL)

            model = AutoModelForCausalLM.from_pretrained(
                                                        MODEL,
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                    ).to(device=device, non_blocking=True)
                        
            model = pipeline(
                        'text-generation', 
                        model=model,
                        tokenizer=MODEL,
                        device= device
                    )

        return model
    
    @torch.no_grad()
    def generate_response(self, input_text, **kwargs):
        
        if "chatgpt" in self.model_config_path: #ChatGPT api does not require tokenizing
            openai.api_key = self.model_config['api_key']
            completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", 
                         "content": "You are a helpful assistant that answers football related questions. Based on references provided, answer the query professionally. You are not allowed to add any fabrication to the answer. You must provide all your answer in Korean. If the user asks non-football related matters, do not answer"},
                        {"role": "user", 
                         "content": input_text}],
                    #plugins=["wikipedia"]
                    #temperature = 0.8

                    )

            response_text = completion["choices"][0]['message']['content']
        else:
            
            output = self.model(
                            input_text,#f"### 질문: {input_text}\n\n### 답변:", 
                            do_sample = self.model_config['do_sample'],
                            max_new_tokens = 512,
                            temperature = self.model_config['temperature'],
                            top_p = self.model_config['top_p'],
                            return_full_text = False,
                            eos_token_id = 2
                            )
            
            response_text = output[0]['generated_text']
            del output
                    
        return response_text
            
    def Request_Rewriter(self, history_rewrite_request, request, **kwargs):
        
        history_text = "".join(history_rewrite_request)

        #If there is no history requests, do not rewrite query.
        if (len(history_rewrite_request) > 0):
        
            query_rewrite_input = global_no_demon_template.format(input=request_rewriting_template.format(history = history_text, request = request))
            request = self.generate_response(query_rewrite_input, **kwargs)
            
            #show the rewritten query on the demo to verficate whether the rewriting is correct.

        print("챗봇에 적합하게 재구성된 입력:" + request)

        return request
    
    def Retrieve_documents(self, query, data_dir, top_k: int = 20, 
                           check_point_dir = None,
                           index_save_dir = None,
                           retrieve_mode = 'sparse'
                           ):
        save_dir = './index_save_dir/'
        os.makedirs(save_dir, exist_ok=True)
        
        if retrieve_mode == 'dense':
                    
            tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
            model = KobertBiEncoder()
            model.cuda()
            model.load(check_point_dir)
            model.eval()
            
            builder = Index_Builder(model = model, 
                            tokenizer=tokenizer,
                            data_dir = data_dir,
                            index_save_dir = save_dir,
                            use_content_type = 'All',
                            batch_size = 64,
                            retrieve_mode = retrieve_mode
                            )
            
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
                data_dir = data_dir,
                index_save_dir = save_dir,
                use_content_type = 'All',
                batch_size = 64,
                retrieve_mode = retrieve_mode
                )
            
            saved_index_path = os.path.join(save_dir,'sparse','bm25_index.pickle')
            #saved_index_path = os.path.join(save_dir,'sparse','bm25result.pickle')

            if os.path.exists(saved_index_path):
                # Load the BM25 index from a file
                with open(saved_index_path, 'rb') as index_file:
                    loaded_bm25 = pickle.load(index_file)
                print("Loading Saved sparse index")
            else:
                loaded_bm25 = builder.build_sparse_index()
            
            raw_reference_list = builder.search(query = query, 
                                                 top_k = top_k,
                                                 searcher = loaded_bm25)

        return raw_reference_list  
        
        
    def Passage_Extractor(self, raw_reference_list, query, cutoff: int,
                           extract_step : int, extract_window : int,
                           if_extract: bool = True, **kwargs
                        ):
        
        '''
        Generate Answer based on retrieved references
        
        '''
        
        key_passg = []
        
        reference = ""
        print('Extracting relevant Sources')
        for idx, raw_reference in enumerate(raw_reference_list):
            
            if if_extract: 
                ref_length = len(raw_reference["contents"])
                
                passage = ""
                
                for j in tqdm(range(0, ref_length, extract_step)) :             
                    extraction_input = passage_extraction_template.format(content = raw_reference["title"] + ":" + raw_reference["contents"][j : j + extract_window], question=query)
                    real_extraction_input = global_no_demon_template.format(input = extraction_input)
                    #real_extraction_input = global_template.format(demon1 = demon1, summary1 = summary1, demon2 = demon2, summary2 = summary2, input = extraction_input)
                    fragments = self.generate_response(real_extraction_input, **kwargs)
                    print(fragments)
                    passage = passage + fragments + "\t"
            else :
                
                extraction_input = passage_extraction_template.format(content = raw_reference["title"] + ":" + raw_reference["contents"], question=query)
                #real_extraction_input = global_no_demon_template.format(input = extraction_input)
                
                try:
                    fragments = self.generate_response(extraction_input, **kwargs)
                    #passage = passage + fragments + "\t"
                    passage = fragments
                    print(passage)
                except:
                    passage = ""
                    
                key_passg.append(passage)
                #passage = raw_reference["contents"]
                
            reference += ('\n Reference {} : '.format(idx+1) + passage)
            input_text = answer_generation_template.format(context = passage, question = query)

            output = self.generate_response(input_text, **kwargs)
            print(output)
            
        reference = reference[:cutoff]

        
        return reference, key_passg

    
    def Answer_Generator(self, reference, revised_request, **kwargs):
        
        """
        Generate Answer based on retrieved references
        """
        
        input_text = answer_generation_template.format(context = reference, question = revised_request)
        #input_text = global_no_demon_template.format(input=input())
        output = self.generate_response(input_text, **kwargs)


        return output
    
    def Fact_Checker(self, reference, output, **kwargs):
        return True
        answer_input = fact_checking_template.format(reference = reference, answer = output)
        input_text = global_no_demon_template.format(input=answer_input)
        output = self.generate_response(input_text, **kwargs)
        
        if ("no" in output or "No" in output or "NO" in output) :
           return True
        if ("yes" in output or "Yes" in output or "YES" in output) :
           return False    
    
