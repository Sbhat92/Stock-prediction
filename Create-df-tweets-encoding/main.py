import os
import json
import pandas as pd
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input):
  return model(input)

# Specify the root directory where your JSON files are located
root_directory = 'preprocessed/'

columns = []
for i in range(512):
     columns.append(f"embedding_{i+1}")

# Walk through all directories and files in the specified root directory
for foldername,subdirectories, filenames in os.walk(root_directory):

        
        rows_list = []
        for filename in filenames:
   
            json_file_path = os.path.join(foldername, filename)
            if filename == '.DS_Store':
                 continue
            if filename == 'sentiments.csv':
                 continue    
                
            # Read the JSON file
            with open(json_file_path, 'r') as file:
                # Load JSON data
                #print("wvsvsrvsvDDCDDDCCCCC",file,json_file_path)
                #print(file,filename,json_file_path)
                val_total = [0]*512
                l=0
                for line in file:
                    l+= 1
                    #print("line",line)
                    msg_dict = json.loads(line)
                    df = []
                    json_data = " ".join(msg_dict['text'])
                    #print(json_data)
                    #SIA=SentimentIntensityAnalyzer()
                    #polarity = SIA.polarity_scores(json_data)['compound']
                    message_embeddings = embed([json_data])

                    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
                        
                        val_total = [sum(x) for x in zip(message_embedding, val_total)]
                for i in val_total:
                    i = i/l
                #print("snippet:,",filename,[val])
                dict1 = [filename] + val_total
                
                rows_list.append(dict1)
        #print(rows_list)
        
        df = pd.DataFrame(rows_list,columns=['Date']+columns)  
        #print(df,foldername)
        df.to_csv(f'{foldername}/sentiments.csv')  