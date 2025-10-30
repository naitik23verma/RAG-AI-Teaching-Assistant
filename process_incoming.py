import requests
import os 
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import joblib


def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding=r.json()['embeddings']
    return embedding

def inference(prompt):
    r=requests.post("http://localhost:11434/api/generate",json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream":False
    })
    response=r.json()
    return response

df=joblib.load("embeddings.joblib")

incoming_query=input("Ask a Question: ")
question_embedding=create_embedding([incoming_query])[0]



np.vstack(df['embedding'].values)

similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# print(similarities)
top_results=5
max_indx = similarities.argsort()[::-1][0:top_results]

new_df=df.loc[max_indx]

# for index,item in new_df.iterrows():
#     print(index,item['topic'],item['timestamp'],item['text'])


prompt=f''' I am teching Web development using Sigma Web Development course Here are video subtitle chunks containing chunks id, video number, video title, timestamp [start,end] in sec, and the text at that time 
{new_df[['chunk_id','topic','timestamp','text']].to_json(orient="records")}

-----------

"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught where (in which video and
what timestamp) and guide the user to go to that particular video.Response should be short and talk just in normal language dont mention chunks or any complex term. if user asked unrelated question tell him you can only answer the question related to this course.
'''    
with open("prompt.txt","w") as f:
    f.write(prompt)


response= inference(prompt)["response"]  
print(response)  
# print(response.json())  
# with open("response.txt","w") as f:
#     f.write(response)
