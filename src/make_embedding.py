from glob import glob
import json
import os
import random
import openai
import pandas as pd
from tqdm import tqdm

from secret import api_key

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

def document_embeddings(documents, api_key, model='text-embedding-ada-002'):
    
    openai.api_key = api_key

    result = []

    for idx, doc in enumerate(tqdm(documents)):
    
        try:
            sample = '제목: '+doc["title"]+' 본문: '+doc["content"]
            output = openai.Embedding.create(input=sample, model=model)
            embedding = output['data'][0]['embedding']
            doc['embedding'] = embedding
            
            with open(f'/Project/embedding_data_wiki/document_and_embedding_{idx}_wiki.json', 'w') as file:
                json.dump(doc, file)
            
            result.append(doc)
        except:
            continue
    
    return result

def calculate_cost(avg_tokens: int, 
                   num_docs : int,
                   cost_per: float) -> int:
    """Returns the cost of a text string in tokens."""
    
    total_cost = num_docs* avg_tokens * cost_per
    
    return total_cost

if __name__ == '__main__':

    docs = load_json_data('/Project/json_data')
    print("Total Cost: ", calculate_cost(1100, len(docs), 0.0004/1000))
    sample_documents = docs #random.sample(docs, 10000)
    embedding_list = document_embeddings(sample_documents, api_key)