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

jsons=os.listdir("jsons")
chunk_id=0
my_dicts=[]
for json_file in jsons:
    topic = os.path.splitext(json_file)[0]
    with open(f"jsons/{json_file}") as f:
        content=json.load(f)
    print(f"Creating Embeddings for {json_file}")    
    embeddings=create_embedding([c['text'] for c in content['chunks']])    
    for i,chunk in enumerate(content['chunks']):
        chunk['chunk_id']=chunk_id
        chunk['embedding']=embeddings[i]
        chunk['topic'] = topic
        chunk_id+=1
        my_dicts.append(chunk)  
           

# a=create_embedding(["I am a boy","My name is naitik"])
# print(a)
df=pd.DataFrame.from_records(my_dicts)

joblib.dump(df,'embeddings.joblib')
