import os
import json
import argparse
import yaml
import warnings
import asyncio

from utils import load_model, generate_answer, Retrieve_documents
from retriever import Index_Builder

from transformers import AutoTokenizer
import pandas as pd

from secret import api_key
import guidance

import gradio as gr

warnings.filterwarnings(action='ignore')

from prompts import refine_query_prompt, confidence_prompt, default_prompt, strategy_ext_prompt, answer_prompt

def load_prompts(prompt_dir):
    # Load txt file
    with open(prompt_dir, 'r', encoding = 'utf-8') as f:
        prompt = f.read()

    return prompt

def main(message, history):
    
    
    #guidance = load_model(args['model_name'], api_key)

    print(message)
    print(history)
    
    args = yaml.load(open("/Project/src/config/runner_chatgpt.yaml",'r'), Loader=yaml.FullLoader)
    
    query = args['query'] = message
    args['idx'] = 0
    history = []

    # refine_query_prompt = load_prompts(args['query_refine_dir'])[0]
    # confidence_prompt = load_prompts(args['confidence_dir'])[0]
    # default_prompt = load_prompts(args['default_dir'])[0]
    # strategy_ext_prompt = load_prompts(args['strategy_ext_dir'])[0]
    # answer_prompt = load_prompts(args['answer_dir'])[0]
    
    #structure_program = guidance(refine_query_prompt, query = query)
    #res_refine = generate_answer(args['model_name'], api_key, refine_query_prompt, query = query)
    
    #refined_query = res_refine['refined_query']
    
    refined_query = query
    sparse_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    
    print('Retrieving documents...')
          
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
       
    print('Generating Answer Now...')

    if len(history)==0:

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
            
        print(answer)
        return answer
    
    else:
        pass

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='FT-Chatbot Operations')
        
    parser.add_argument('--save_dir', type=str, default=None, help='Data directory')

    args = parser.parse_args()
        
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    demo = gr.ChatInterface(
        main,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="축구 관련 질의를 해주세요", container=False, scale=7),
        title="[FT-LLM] 축구 관련 질의응답 챗봇 서비스 (Demo)",
        description=f'''
❗️ BK21 4단계 사업의 데이터 분석 연구과제를 위한 대화형 추천 시스템(demo)입니다. 
본 챗봇은 사용자의 학술적 관심분야를 분석하여 지도교수(연구실)를 체계적으로 추천하는 대화형 시스템입니다. 
사용자가 자신의 관심 있는 학술 분야에 대해 상세히 작성하면, 더욱 정밀한 지도교수(연구실)추천이 가능합니다.
* 사용 가능 학과: 공과대학, 생명과학대학, 정보대학, 정경대학(통계학과)
''',
        theme="JohnSmith9982/small_and_pretty",
        examples=[
            "나는 이미지와 비디오 데이터에서 패턴을 인식하고 예측하는 모델을 개발하는데 관심이 많아. 특히, 딥러닝과 컴퓨터 비전을 사용하여 의료 이미지 데이터를 분석하는 연구를 하고 싶어.", 
            "나는 태양광 및 풍력 에너지 시스템의 효율성을 높이는 방법에 대해 연구하고 싶어. 특히, 에너지 저장 솔루션과 스마트 그리드 기술에 관심이 있어.", 
            "나는 기후 변화의 영향을 평가하고, 지속 가능한 개발을 위한 전략을 모색하는 데 관심이 있어. 특히, 대기 및 수질 오염을 모니터링하고 감소시키는 기술에 집중하여 연구를 하고 싶어."
            ],
        cache_examples=False,
        retry_btn=None,
        # undo_btn="Delete Previous",
        # clear_btn="Clear",
    )
    
    demo.launch(share=True)
