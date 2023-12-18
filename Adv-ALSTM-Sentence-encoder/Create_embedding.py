import numpy as np
import pandas as pd
import os
import glob

columns = []
for i in range(512):
     columns.append(f"embedding_{i+1}")

root_directory = "preprocessed/"
for foldername,subdirectories, filenames in os.walk(root_directory):
    path = f"{foldername}/*.csv"
    for fname in glob.glob(path):
        x = (foldername).split('/')[1]
        
        df = pd.read_csv(fname)
        #df.set_index('Date')
        df = df[['Date']+columns]
        if x != "":
            df2 = pd.read_csv(f'data/stocknet-dataset/price/raw/{x}.csv')
            print(df2)
            #df['Date'] = df['Date'].astype(str)



            df3 = df2.merge(df,on='Date',how='left')
            
            df3= df3.fillna(0)
            print(df3)
            df3.to_csv(f'data_sentiment/{x}.csv')
            