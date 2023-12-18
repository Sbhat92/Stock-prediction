### Stock price prediction using tweets    
The financial domain heavily rely on the stock movement prediction task. Stock movement prediction is a well studied task. Stock forecasting is complex, given the stochastic dynamics and non-stationary behavior of the market. Stock movements are influenced by varied factors, that are hard to model, and often interact with each other. The rising effect of the internet, and social media such as Twitter on stock prices must be modelled along with historic prices to encapsulate it's effect. We introduce an architecture that achieves a potent blend of chaotic temporal signals from financial data and social media. Through experiments on real-world S&P 500 index data and English tweets, we show the practical applicability of our model as a tool for investment decision making and trading.

To run the code, use the data from: https://github.com/yumoxu/stocknet-dataset, add it to the current directory in the data/ folder. To run any variant of the ALSTM model, you will have to create embeddings for tweets. The code for this is in the Create-df-tweets directories of respective methods. The data for each method can be referenced in the report. 

Upon creating the embedding, run `preprocessing.py` in the method of choice, then run `main.py` to train the model.

Requirements:

Python 3.7,
Tensorflow2,
Numpy,
pandas,
pickle,
gensim
