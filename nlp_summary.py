import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize

import networkx as nx


if __name__ == "__main__":
    nltk.download('punkt')

class summarise:
    def __init__(self):
        self._load_stopwords()
        #self._word_embeddings_func("./data/glove.6B.100d.txt")

    
    def _load_stopwords(self):
        self.stop_words = stopwords.words("english")
        

    def read_article(self, filedata):
        article = filedata.split(". ")
        sentences = []

        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 

        return sentences
        
        
    def _preprocess(self, series):
        self.sentences = series
        
        self.sentences = self._replace_rank(self.sentences)
        self.sentences = self._tokenize_to_sentence(self.sentences)
        
        #replace with function
        #self.sentences = [[text.lower() for text in article] for article in self.sentences]        
        self.sentences = [[text.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ") for text in article] for article in self.sentences]
        
        
        
        self.sentences = self._tokenize_to_words(self.sentences)
        
        #self.sentences = self._remove_stopwords(self.sentences)
        #self.sentence_vectors = self._vectors(self.sentences)
        #self.summary = self._perform_summarisation(self.sentences)
        

    def _tokenize_to_sentence(self, articles):
        if articles is not pd.Series: articles = pd.Series(articles)
        
        sentences = []
        for s in articles:
            sentences.append(sent_tokenize(s))
        return sentences
    
    def _tokenize_to_words(self, articles):
        if articles is not pd.Series: articles = pd.Series(articles)
        
        sentences = []
        for text in articles:
            art = []
            for sentence in text:
                art.append(sentence.split(" "))
                #art.pop()
            sentences.append(art)
        return sentences

    
    def _replace_rank(self, articles):
        pattern = "(No\. [0-9])|(No\.[0-9])"
        pattern_1 = "No\. [0-9]"
        pattern_2 = "No\.[0-9]"
        
        
        if articles is not pd.Series: articles = pd.Series(articles)
        sentences = []
        
        for string in articles:     
            it = 0
            while True:
                result = re.search(pattern, string)
                if result is None: break
                result_1 = re.search(pattern_1, string)
                if result_1 is not None:
                    loc = result_1.start()
                    string = string[0:loc] + "No" + string[loc+4:]
                result_2 = re.search(pattern_2, string)
                if result_2 is not None:
                    loc = result_2.start()
                    string = string[0:loc] + "No" + string[loc+3:]
                it += 1
                if it > 500: break
            sentences.append(string)
        return sentences

    def _sentence_similarity(self, sentance_1, sentance_2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sentance_1 = [w.lower() for w in sentance_1]
        sentance_2 = [w.lower() for w in sentance_2]

        all_words = list(set(sentance_1 + sentance_2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sentance_1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sentance_2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)
    

    def _build_similarity_matrix(self, sentences):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], self.stop_words)
        return similarity_matrix
    
    def _generate_summary(self, text, top_n, split):
        
        sentence_similarity_martix = self._build_similarity_matrix(text)

        
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)


        l = pd.Series(scores).sort_values(ascending=False).index[:top_n].sort_values()
        sum_text = []
        for i in range(min(len(text), top_n)):
            sum_text.append(" ".join(text[l[i]]))
        output = split.join(sum_text)
        return output
        
    
    
    def generate_summary(self, articles, top_n=3, split = "\n\n"):
        
        self._preprocess(articles)
        
        outputs = []
        for text in self.sentences:
            outputs.append(self._generate_summary(text, top_n = top_n, split = split))
                
        return outputs