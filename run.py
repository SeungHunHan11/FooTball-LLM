from model.model import load_models
import os
import json
import argparse
import yaml
import warnings

warnings.filterwarnings(action='ignore')

def main(src, solver, config):
    
    query, history, history_rewrite_request, history_url = src
    
    print('최초 입력: ', query)
    new_query = solver.Request_Rewriter(history_rewrite_request = history_rewrite_request, request = query)

    history_rewrite_request.append(new_query)
    
    print('Checking relevant Sources')
    raw_lists = solver.Retrieve_documents(
                                query = new_query, 
                                data_dir = config['Retriever_config']['data_dir'], #"./json_data/", 
                                top_k = config['Retriever_config']['top_k'],
                                check_point_dir = config['Retriever_config']['Retriever_pt'], #'/Project/2050iter_model.pt',
                                index_save_dir = config['Retriever_config']['index_dir'],
                                retrieve_mode = config['Retriever_config']['retrieve_mode']
                                )
    
    #collect the retrieved urls
    source = []
    urls = []
    title = []
    for raw_reference in raw_lists :
        try:
            urls.append(raw_reference['url'])
            title.append(raw_reference['title'])
            source.append("- {}: {}".format(raw_reference['title'], raw_reference['url']))
        except:
            urls.append('No URL')
            title.append(raw_reference['title'])
            source.append("- {}: {}".format(raw_reference['title'], 'No URL'))

    #recorde all historical retrieved urls
    history_url.append(urls) 
    
    reference, key_passg = solver.Passage_Extractor(raw_reference_list = raw_lists, query = new_query, 
                                         cutoff = config['Retriever_config']['cutoff'], 
                                         extract_step = config['Retriever_config']['extract_step'], 
                                         extract_window = config['Retriever_config']['extract_window'], 
                                         if_extract = config['Retriever_config']['if_extract'],
                                         index_dir = config['Retriever_config']['index_dir']
                                         )
    
    # if len(history) > 0:
    #     for i, (ques, ref, res) in enumerate(history):
    #         res = res + '\n' +"Reference URL:\n" + ref[i]
    #         print("이전 사용자의 질문: ", ques, end = '\n')
    #         print("이전 답변: ", res, end = '\n\n')
    
    response = solver.Answer_Generator(reference = reference, revised_request = new_query)
    
    fact_check = solver.Fact_Checker(reference = reference, output = response)
    
    if not fact_check: #if the answer is incorrect
        response = '대답할 수 없는 질문입니다. 다시 질문해주세요.'

    print("챗봇: ", response)
    
    history.append((query, reference, response))

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = 'dialogue_'+query+'.json'

    url_dict = {}
    for i in range(len(urls)):
        name = f'url_{i}'
        url_dict[name] = 'Relevant Passage: Hint: ' + key_passg[i] + 'title: '+ title[i] + " url: " + urls[i]

    with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
        f.write(json.dumps((
                        {'Query':query, 
                         'response': response, 
                         "urls" : url_dict}),
                           ensure_ascii=False,
                           indent = '\t'
                           ))

    return history, history_rewrite_request, history_url

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='FT-Chatbot Operations')
    
    parser.add_argument('--yaml_config', type=str, default=None, help='Chat config file')    
    args = parser.parse_args()
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    
    history = []
    history_rewrite_request = []
    history_url = []
    
    solver = load_models(model_config_path = cfg['Model_config']['llm_configs'],#"./config/llm_chatgpt.json", 
                        device = cfg['Model_config']['device']#"cuda:1"
                        )
    
    while True:
        query = input('Enter any inquries! (quit to exit):')

        if query == 'quit':
            print('Chat Quitted')
            break

        src = query, history, history_rewrite_request, history_url
        
        history, history_rewrite_request, history_url = main(src, solver, cfg)