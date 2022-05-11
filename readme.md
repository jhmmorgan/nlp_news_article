# Natural Langugage Processing
## Summarising News Articles, warning users if the article contains hate speech or fake news.

**James Morgan (jhmmorgan)**

_2022-05-11_

## 📖 Background

We want a proof of concept, where an end user can easily be provided with a summary of a news article, along with a warning on whether the text is likely to contain hate speech or fake news.

This proof of concept would be in the form of a standalone application that when provided the URL to a news article, provides the end-user with the summary of the article, along with a flag if the article may contain hate speech or fake news.

---

## 🔬 Approach
This project has been split into four parts showcasing custom classes and trained models.  
 - [Part 1 - Fake News and Hate Speech](https://github.com/jhmmorgan/nlp_news_article/blob/main/NLP%201%20-%20Fake%20and%20Hate%20News.ipynb)
 - [Part 2 - Summarising articles](https://github.com/jhmmorgan/nlp_news_article/blob/main/NLP%202%20-%20Summarising%20articles.ipynb)
 - [Part 3 - Web Scraping](https://github.com/jhmmorgan/nlp_news_article/blob/main/NLP%203%20-%20Web%20Scraping.ipynb)
 - [Part 4 - Bringing it all together](https://github.com/jhmmorgan/nlp_news_article/blob/main/NLP%204%20-%20Final%20Outcome.ipynb)

---

## 📚 The final outcome
The final outcome of this project is a custom class, which produces a summary of any article of text along with warning flags to indicate if the article might contain fake news or hate speech.
```python
from nlp_article_summary import *

url = "https://www.dailymail.co.uk/news/article-10759651/Ukraine-war-Putin-suggest-use-nukes-necessary.html"
nlp = NLP_summary()
nlp.process_article_from_url(url)
```

---

## 🎓 Summary and next steps
<div class="alert alert-block alert-info">
<b>How I can improve this project in the future?</b>
</div>


The models aren't perfect.  
1. We could train our fake news and hate speech models in a more shopisticated way
    - There's no training around the source of articles
    - There's no training around key hate words
    - There's no understanding of context, which could turn any normal phrase into hate news when used in certain ways.
2. The summarisation technique is simple but effective and it'll take significantly more effort to create an abstrative summarisation technique.
3. The news article scraper is good, however for the proof of concept we've taken a manual approach of extracting text.
     - We could use third-party libraries, or train a model to identify key tags to extract text from
4. Our predictions aren't perfect.
    - We trained our models on limited data
    - The hate speech model was based on tweets, which contain a lot fewer words than a news article, which likely impacts the performance
    - So whilst our models perform great on their trained data, this may not translate perfectly to articles of text of different sizes or of different context.


That said, as a proof of concept, I consider this project a success.


<div class="alert alert-block alert-info">
<b>Next Steps</b>
</div>

One thing missing from this project is a nice front end to run my models.  It would be nice to create a HTML front-end where I can deploy this model for anyone to use.