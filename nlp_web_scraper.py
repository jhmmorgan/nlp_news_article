# importing the necessary packages
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import time
import re
import urllib

class scrape_article_text:
    def __init__(self, url):
        self.url = url
        self._news_attributes()
        
    def get_article(self):
        return self._scrape_url(self.url)
    
    def _scrape_url(self, url):
        article = requests.get(url)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, "html.parser")
        
        
        domain        = self._get_domain(url)
        domain_att    = self._news_items.get(domain)
        final_article = self._news(soup_article, domain_att)

        # Removing special characters
        final_article = re.sub("\\xa0", " ", final_article)
        final_article = re.sub("  ", " ", final_article)
        
        self.final_article = final_article
        return final_article
    
        
    def _news(self, article, domain_att):
        if domain_att is None:
            soup_text = article.get_text()
            soup_text = re.sub("\n", " ", soup_text)
            return soup_text

        result = []
        if (len(domain_att) > 1):
            unwanted = article.find_all(**domain_att[1])
            for x in unwanted:
                x.extract()
            
        result.append(article.find_all(**domain_att[0]))      

        # Unifying the paragraphs
        list_paragraphs = []
        for body in result:
            for p in np.arange(0, len(body)):
                paragraph = body[p].get_text()
                paragraph = re.sub('\n', '', paragraph)
                paragraph = re.sub('\.', '. ', paragraph)
                list_paragraphs.append(paragraph)
                final_article = " ".join(list_paragraphs)
        
        return final_article
    
    def _get_domain(self, url):
        
        parsed_url = urllib.request.urlparse(url)
        domain = parsed_url.netloc
        domain = re.sub("www.", "", domain)
        domain = re.sub("en.", "", domain)
        domain = re.sub("news.", "", domain)
        domain = re.split("\.", domain)[0]
        
        return domain
        
        
    def _news_attributes(self):
        # {url : [{dict_include}, *{dict_exclude}]}
        self._news_items = {"dailymail"   : [{"name" : "p", "attrs" : "mol-para-with-font"}],
                            "theguardian" : [{"id" : "maincontent"}],
                            "sky"         : [{"class_" : "sdc-article-body--story"}, 
                                             {"class_" : ["sdc-site-video", "sdc-article-related-stories", "sdc-site-au"]}]}

