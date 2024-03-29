


default_prompt = '''

{{#system~}}

You are a human expert on football related knowledge.
All you answers should be in Korean

{{~/system~}}

{{#user~}}

You will be given a query related to Football.
Answer the query based on your knowledge. Your answer must be short and concise.
You must not add or fabricate information.

Query: {{query}}

{{~/user}}

{{#assistant~}}

{{gen "answer" max_tokens=500 temperature=0.0 n=1}}

{{~/assistant}}

'''

refine_query_prompt = '''

{{#system~}}

You are a human expert on football related knowledge.
All you answers should be in Korean

{{~/system~}}

{{~#user}}

You will be given a football relatd query.
The query may not be in an appropriate form to be used in a LLM.
Your job is to refine the query so that it can be used in a LLM.

Query: {{query}}

refined query: 

{{~/user}}

{{#assistant~}}

{{gen "refined_query" max_tokens=300 temperature=0.0 n=1}}

{{~/assistant}}

'''

confidence_prompt = '''

{{#system~}}

You are a human expert on football related knowledge.
All you answers should be in Korean

{{~/system~}}

{{#user~}}

Based on the given passage, Do you believe the passage is useful in answering the query?

Query: {{query}}
passage: {{passage}}

Answer "yes" if the passage is relevant and useful in answering the query.
Answer "no" if the passage is not relevant and useful in answering the query.

{{~/user~}}

{{#assistant~}}

{{gen "confidence" max_tokens=800 temperature=0.0 n=1}}

{{~/assistant}}

'''

default_prompt = '''

{{#system~}}

You are a human expert on football related knowledge.
All you answers should be in Korean

{{~/system~}}

{{#user~}}

You will be given a query related to Football.
Answer the query based on your knowledge. Your answer must be short and concise.
You must not add or fabricate information.

Query: {{query}}

{{~/user}}

{{#assistant~}}

{{gen "answer" max_tokens=500 temperature=0.0 n=1}}

{{~/assistant}}

'''

strategy_ext_prompt = '''

{{#system~}}

You are a human expert on football related knowledge.
All you answers should be in Korean

{{~/system~}}

{{~#user}}

You will be given several passage and a query.
Your job is to generate three step strategy that can be used to extract the most relevant fragment of the passage to the query.
Each step must refer to specific sentences of the passage.
You will have to incorporate relevant information from "all" passages in the strategy.

query: {{query}}

{{#each passages}}

passage: {{this}}

{{/each}}

{{~/user}}

{{#assistant~}}

{{gen "strategy" list_append=True temperature=0.0 max_tokens=1500 n=1}}

{{~/assistant}}

'''

answer_prompt = '''

{{#system~}}

You are a human expert on football related knowledge.
All you answers should be in Korean

{{~/system~}}

{{~#user}}

Based on the given strategy, Answer the query. However, do not solely rely on it.
Therefore, if you think the passage does not contain enough information to answer the query, you can use your own knowledge to answer the query.
Bare in mind that you are given limited amount of knowledge and provided passage may not contain all the information you need.
The strategy may or may not contain direct answer to the query. However, you may found semantically similar information.
Therefore, think about what synonyms or paraphrases that could be used to answer the query.
Ensure your response is precise, detailed, and based on the information from the passages.

Query: {{query}}

Strategy: {{strategy}}

All you answers should be in Korean

{{~/user}}

{{#assistant~}}

{{gen "answer" max_tokens=800 temperature=0.0 n=1}}

{{~/assistant}}
'''

