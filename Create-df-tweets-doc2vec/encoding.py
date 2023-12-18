import gensim.models as g

model="model_folder/doc2vec.bin"  #point to downloaded pre-trained doc2vec model

#load model
m = g.Doc2Vec.load(model)