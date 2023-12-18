import os
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.downloader.download('vader_lexicon')
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
# Specify the root directory where your JSON files are located
root_directory = 'preprocessed/'



# Walk through all directories and files in the specified root directory
for foldername,subdirectories, filenames in os.walk(root_directory):

        
        rows_list = []
        for filename in filenames:
   
            # Check if the file has a JSON extension
            
            # Construct the full path to the JSON file
            json_file_path = os.path.join(foldername, filename)

            # Read the JSON file
            with open(json_file_path, 'r') as file:
                # Load JSON data
                for line in file:
                    msg_dict = json.loads(line)
                    df = []
                    json_data = " ".join(msg_dict['text'])
                    SIA=SentimentIntensityAnalyzer()
                    polarity = SIA.polarity_scores(json_data)['compound']
                    #df['Polarity Score']=df["text"].apply(lambda x:SIA.polarity_scores(x)['compound'])
                    #df["Date_Parsed_final"]=[(df['Date_Parsed'][i].strftime("%Y-%m-%d")) for i in range(len(df))]
                    # Now you can work with the JSON data as needed
                    
                    dict1 = [filename,polarity]
                    

                    rows_list.append(dict1)
        df = pd.DataFrame(rows_list, columns=['Date', 'Polarity'])  
        print(df,foldername)
        df.to_csv(f'{foldername}/sentiments.csv')  