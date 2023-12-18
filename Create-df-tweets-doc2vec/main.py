import os
import json
import pandas as pd
from absl import logging
import pickle
import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
#model = hub.load(module_url)

# def embed(input):
#   return model(input)

# Specify the root directory where your JSON files are located
root_directory = 'preprocessed/'

with open('gensim.pkl', 'rb') as file:  
    model = pickle.load(file)

columns = ['embedding1','embedding2','embedding3','embedding4']


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
                val_total = [0]*4
                l=0
                for line in file:
                    l+= 1
                    #print("line",line)
                    msg_dict = json.loads(line)
                    df = []
                    json_data = " ".join(msg_dict['text'])
                    #print(json_data)
                    sentence = msg_dict['text']
                    #SIA=SentimentIntensityAnalyzer()
                    #polarity = SIA.polarity_scores(json_data)['compound']
                    message_embeddings = model.infer_vector(sentence)
                    #print(message_embeddings)
                    
                    val_total = [x + y for x, y in zip(val_total, message_embeddings)]
                for i in val_total:
                    i = i/l
                #print("snippet:,",filename,[val])
                dict1 = [filename] + val_total
                
                rows_list.append(dict1)
        #print(rows_list)
        
        df = pd.DataFrame(rows_list,columns=['Date']+columns)  
        #print(df,foldername)
        df.to_csv(f'{foldername}/sentiments.csv')  