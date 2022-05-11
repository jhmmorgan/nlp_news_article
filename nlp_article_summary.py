from utils import *
from nlp_models import *
from nlp_summary import *
from nlp_web_scraper import *


class NLP_summary:
    def __init__(self, url = None, top_n = 3):
        self.split = "\n\n"
        self.top_n = top_n
        self._load_all_models()
        
        if url is not None:
            self.process_article_from_url(url)
    
    def process_article_from_url(self, url, return_all = False, print_result = True, top_n = None):
        if top_n is not None: self.top_n = top_n
        sat     = scrape_article_text(url)
        article = sat.get_article()
        output  = self.process_articles(article, return_all = return_all, print_result = print_result)
        if return_all is True: return output        
    
    def process_articles(self, articles, return_all = False, print_result = True):
        if type(articles) == str: articles = pd.Series(articles)
        
        self.is_hate = self._dhs.predict(articles)
        self.is_fake = self._dfn.predict(articles)
        self.summary = self._summarise_text.generate_summary(articles, top_n = self.top_n, split = self.split)
        
        if print_result is True: self.print_output()
        
        if return_all is True: return self.is_hate, self.is_fake, self.summary

    def print_output(self):
        if self.is_hate[0] == 0:
            print2.highlight("Content unlikey to include hate speech.")
        else:
            print2.warning("Content might to include hate speech.   ")
            
        if self.is_fake[0] == 0:
            print2.highlight("Content unlikey to be fake news.       ")
        else:
            print2.warning("Content might to be fake news.          ")

        print("")
        print2.heading("Summary of article")
        print(self.summary[0])

    def _load_all_models(self):
        self._load_dhs()
        self._load_dfn()
        self._load_summariser()
    
    def _load_summariser(self):
        self._summarise_text = summarise()
        
    def _load_dhs(self):
        self._dhs = self._load_model("./models/hate_speech_v1.pkl")
    
    def _load_dfn(self):
        self._dfn = self._load_model("./models/fake_news_v1.pkl")
        
    def _load_model(self, filepath):
        model = nlp_model()
        model.load_model(filepath)
        return model