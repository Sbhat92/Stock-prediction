from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from nltk.tokenize import word_tokenize
import pandas as pd
import sys
import pandas as pd
import smart_open

import pickle


tweets = pd.read_csv("tweets_remaining_09042020_16072020.csv", delimiter=';',quoting=0, header = None)
#print(tweets[2])
tw = list(tweets[2])
print(len(tw))
corpus = " ".join(tw)

with open('corpus.cor', 'w') as f:
    f.write(corpus)

# def build_model(max_epochs, vec_size, alpha, tag_data):
    
#     model = Doc2Vec(vector_size=vec_size,
#                alpha=alpha,
#                min_alpha=0.00025,
#                min_count=1,
#                dm=1)
    
#     model.build_vocab(tag_data)
    
#     # With the model built we simply train on the data.
    
#     for epoch in range(max_epochs):
#         print(f"Iteration {epoch}")
#         model.train(tag_data,
#                    total_examples=model.corpus_count,
#                    epochs=model.epochs)

#         # Here I decrease the learning rate. 

#         model.alpha -= 0.0002

#         model.min_alpha = model.alpha
    
#     # Now simply save the model to avoid training again. 
    
#     model.save("COVID_MEDICAL_DOCS_w2v_MODEL.model")
#     print("Model Saved")
#     return model

model = gensim.models.doc2vec.Doc2Vec(vector_size=4, min_count=2, epochs=40)

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus('corpus.cor'))

model.build_vocab(train_corpus)

#xprint(f"Word 'penalty' appeared {model.wv.get_vecattr('apple', 'count')} times in the training corpus.")

vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print(vector)


# save the iris classification model as a pickle file
model_pkl_file = "gensim.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)