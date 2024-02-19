import os
import json
import argparse
import yaml
import warnings

<<<<<<< HEAD
import time
from utils import generate_answer, Retrieve_documents
=======
from utils import load_model, generate_answer, Retrieve_documents
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b
from retriever import Index_Builder

from transformers import AutoTokenizer
import pandas as pd

from secret import api_key
import guidance
warnings.filterwarnings(action='ignore')

from prompts import refine_query_prompt, confidence_prompt, default_prompt, strategy_ext_prompt, answer_prompt

def load_prompts(prompt_dir):
    # Load txt file
    with open(prompt_dir, 'r', encoding = 'utf-8') as f:
        prompt = f.read()

    return prompt

def main(args):
    
<<<<<<< HEAD
    current_time = time.time()

    os.makedirs(args['save_dir'], exist_ok = True)
    
    pre_searched = '/Project/src/outputs/revision/GPT-4_Embedding_News_Confidence/{}.json'.format(str(args['idx']))
    
    with open(pre_searched, "r", encoding = "utf-8-sig") as f:
        searched_doc = json.load(f)
        
    searched_doc['Confidence'] = 'No'
    
    if searched_doc['Confidence']=='yes':
        
        save = searched_doc
        # save json file        
        with open(os.path.join(args['save_dir'],str(args['idx'])+".json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                            (save), 
                            ensure_ascii=False,
                            indent = '\t'
                            )
                    )        
        return None
    
    else: 

        if args['query'] is not None:
            query = args['query']   

        else:
            query = input("Please input your question: ")

        if args['use_refine'] == "refined":
            structure_program = guidance(refine_query_prompt, query = query)
            res_refine = generate_answer(args['model_name'], api_key, refine_query_prompt, query = query)
            refined_query = res_refine['refined_query']

        else:
            refined_query = query

        
        # raw_reference_list = []
            
        raw_reference_list, raw_contents = Retrieve_documents(
                                        query = query, 
                                        data_dir = args['data_dir'],
                                        embedding_data_dir = args['embedding_data_dir'],
                                        use_content_type = args['content_type'],
                                        top_k = args['top_k'], 
                                        index_save_dir = args['index_save_dir'],
                                        retrieve_mode = args['retrieve_mode'] #'sparse'
                                        )
                                
        print(raw_reference_list)                  
            
        # try: 
        #     res_confidence = generate_answer(args['model_name'], api_key, confidence_prompt, query = refined_query, 
        #                         passage = [raw_contents[1]]) #has to be a list
        #     confidence = res_confidence['confidence'].lower()

        # except:
        #     first_half = raw_contents[1][:int(len(raw_contents[1])*0.3)]
        #     last_half = raw_contents[1][int(len(raw_contents[1])*0.3):]        

        #     res_confidence = generate_answer(args['model_name'], api_key, confidence_prompt, query = refined_query, 
        #                                     passage = [first_half]) #has to be a list
            
        #     confidence = res_confidence['confidence'].lower()

        confidence = 'yes'
        
        if confidence == 'no':
        
            res_default = generate_answer(args['model_name'], api_key, default_prompt, query = refined_query)        
            
            strategy = 'default'
            answer = res_default['answer']
            
        else:
            
            try:
                res_strategy = generate_answer(args['model_name'], api_key, strategy_ext_prompt, query = refined_query, 
                                            passages = [raw_contents[2]]) # has to be a list
                strategy = res_strategy['strategy']

            except: # Split Passage into two when given passage is too long
                
                first_half = raw_contents[2][:int(len(raw_contents[2])*0.3)]
                last_half = raw_contents[2][int(len(raw_contents[2])*0.3):]      

                res_strategy = generate_answer(
                                            args['model_name'], 
                                            api_key, 
                                            strategy_ext_prompt, 
                                            query = refined_query, 
                                            passages = [first_half]) # has to be a list
                
                strategy = res_strategy['strategy']
                
            print(strategy)
            
            res_answer = generate_answer(args['model_name'], api_key, answer_prompt, query = refined_query,
                                        strategy = strategy) # has to be a list

            first_answer = res_answer['answer']
                                    
            answer = first_answer
            
        save = {'Question': query, 
                'Refined_query': refined_query,
                'Confidence': confidence,
                'Strategy': strategy,
                'Reference': raw_reference_list,
                'Answer': answer,
                'Duration': time.time() - current_time
                }    
                
        # save json file        
        with open(os.path.join(args['save_dir'],str(args['idx'])+".json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                            (save), 
                            ensure_ascii=False,
                            indent = '\t'
                            )
                    )
=======
    
    #guidance = load_model(args['model_name'], api_key)

    if args['query'] is not None:
        query = args['query']   

    else:
        query = input("Please input your question: ")
        
    
    # refine_query_prompt = load_prompts(args['query_refine_dir'])[0]
    # confidence_prompt = load_prompts(args['confidence_dir'])[0]
    # default_prompt = load_prompts(args['default_dir'])[0]
    # strategy_ext_prompt = load_prompts(args['strategy_ext_dir'])[0]
    # answer_prompt = load_prompts(args['answer_dir'])[0]
    
    structure_program = guidance(refine_query_prompt, query = query)
    res_refine = generate_answer(args['model_name'], api_key, refine_query_prompt, query = query)
    
    refined_query = res_refine['refined_query']

    sparse_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    
    
    raw_reference_list = Retrieve_documents(
                            query = query, 
                            data_dir = args['data_dir'],
                            sparse_tokenizer = sparse_tokenizer,
                            use_content_type = args['content_type'],
                            top_k = args['top_k'], 
                            check_point_dir = args['check_point_dir'],
                            index_save_dir = args['index_save_dir'],
                            retrieve_mode = args['retrieve_mode'] #'sparse'
                            )
                            
    print(raw_reference_list)                   
           
    # res_confidence = generate_answer(args['model_name'], api_key, confidence_prompt, query = refined_query, 
    #                                 passage = [raw_reference_list[0]]) #has to be a list
    
    # confidence = res_confidence['confidence'].lower()
    
    confidence = 'yes'
    
    if confidence == 'no':
    
        res_default = generate_answer(args['model_name'], api_key, default_prompt, query = refined_query)        
        
        strategy = 'default'
        answer = res_default['answer']
        
    else:
        
        try:
            res_strategy = generate_answer(args['model_name'], api_key, strategy_ext_prompt, query = refined_query, 
                                        passages = [raw_reference_list[0]]) # has to be a list
            
            strategy = res_strategy['strategy']
            
        except: # Split Passage into two when given passage is too long
            
            try: 
                first_half = raw_reference_list[0]['contents'][:int(len(raw_reference_list[0])*0.5)]
                last_half = raw_reference_list[0]['contents'][int(len(raw_reference_list[0])*0.5):]
            
            except:
                first_half = raw_reference_list[0]['content'][:int(len(raw_reference_list[0])*0.5)]
                last_half = raw_reference_list[0]['content'][int(len(raw_reference_list[0])*0.5):]
            
            
            res_strategy = generate_answer(args['model_name'], api_key, strategy_ext_prompt, query = refined_query, 
                                passages = [first_half]) # has to be a list
            
            strategy = res_strategy['strategy']
            
            # res_strategy_2 = generate_answer(args['model_name'], api_key, strategy_ext_prompt, query = refined_query, 
            #         passages = [last_half]) # has to be a list
            
            # strategy_2 = res_strategy_2['strategy']
            
            # strategy = strategy + '\n' + strategy_2
            
        print(strategy)
        
        res_answer = generate_answer(args['model_name'], api_key, answer_prompt, query = refined_query,
                                     strategy = strategy) # has to be a list

        first_answer = res_answer['answer']
        #final_answer = res_answer['final_answer']
                                
        answer = first_answer
        
    save = {'Question': query, 
            'Confidence': confidence,
            'Strategy': strategy,
            'Reference': raw_reference_list,
            'Answer': answer}    
    
    os.makedirs(args['save_dir'], exist_ok = True)
    
    # save json file        
    with open(os.path.join(args['save_dir'],str(args['idx'])+".json"), 'w', encoding='utf-8') as f:
         f.write(json.dumps(
                        (save), 
                        ensure_ascii=False,
                        indent = '\t'
                        )
                 )
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b
        
    return answer

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='FT-Chatbot Operations')
        
<<<<<<< HEAD
    parser.add_argument('--yaml_config', type=str, default='./config/runner_chatgpt.yaml', help='Chat config file')
    parser.add_argument('--data_dir', type=str, default='./QA_dataset.csv', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./outputs/test', help='Data directory')

    args = parser.parse_args()
    
    if args.data_dir is not None:
        df = pd.read_csv(args.data_dir)
        queries = df['Question']
        
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    for idx, query in enumerate(queries):

        #try:
            # if idx == 1:
            #     break 
        print(query)
        cfg['query'] = query
        cfg['idx'] = idx+31
        answer = main(cfg)
    #    except:
     #       continue
=======
    parser.add_argument('--yaml_config', type=str, default=None, help='Chat config file')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Data directory')

    args = parser.parse_args()
    
    
    if args.data_dir is not None:
        df = pd.read_csv(args.data_dir)
        queries = [df['Question'][18]]
        
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    cfg['save_dir'] = args.save_dir
    
    for idx, query in enumerate(queries):
        
        # if idx == 1:
        #     break 
        print(query)
        cfg['query'] = query
        cfg['idx'] = idx
        answer = main(cfg)
>>>>>>> 8e228dc73cc39899f044884d150e3e627ac5db4b

        #df.loc[idx, 'FT-LLM_YesConfidence_plus_wiki'] = answer
        #df.loc[idx, 'FT-LLM'] = answer
        #df.to_csv(args.data_dir, index = False)
        
        
    