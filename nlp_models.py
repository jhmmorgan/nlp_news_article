from utils import *
import pandas as pd
import joblib
import re
from sklearn.utils import resample

from sklearn.metrics import confusion_matrix


class extensions:
    def save(obj, file):
        joblib.dump(obj, file)
        
    def load(file):
        return joblib.load(file)
    
class nlp_model:
    """
    Class to fit and predict text. Used as a template for future models.
    Both the fit and predict functions will preprocess text by cleaning text or a pandas data series.
    
    Class also allows functionality to resample training data.
    
    Usage:
     > model = model_template()
     > model.fit(X_train, y_train)
     > y_predict = model.predict(X_test)
     
     SAVING/LOADING:
     > model.save_model("model.pkl")
     > model.load_model("model.pkl")
     NOTE: This saves and loads the sklearn pipeline model. It doesn't save this class, which should still be imported.
    
     RESAMPLE
     > df_resampled = model.resample(df, "label")
    """
    def __init__(self, pipeline = None):
        self.model = None
        self._pipeline = pipeline
        self.random_state = 42
        
    def fit(self, X, y):
        if self._pipeline is not None: 
            X = self._preprocessing(X)
            self.model = self._pipeline.fit(X, y)
    
    def predict(self, X):
        if self.model is not None:
            X = self._preprocessing(X)
            return self.model.predict(X)

    def _preprocessing(self, X):
        """Preprocessing that is applied to features(X), whenever Fit or Predict is called."""
        X = self._clean_text(X)     
        return X
        
    def save_model(self, file):
        extensions.save(obj = self.model, file = file)
    
    def load_model(self, file):
        self.model = extensions.load(file)
        
    def resample(self, df, label):
        """
        Function to resample a dataframe so all labels/targets are of equal size.
        It'll find the label with the majority and resample all other labels to be of the same size.
        """
        majority_label = df[label].value_counts().idxmax()
        required_size  = df[label].value_counts()[majority_label]

        df_majority    = df[df[label] == majority_label]
        df_minority    = df[df[label] != majority_label]

        df_resampled   = df_majority
        for i in df_minority.label.value_counts().index:
            df_label     = df_minority[df_minority[label] == i]

            df_sample    = resample(df_label, 
                                    replace      = True,    
                                    n_samples    = required_size,   
                                    random_state = self.random_state)
            df_resampled = pd.concat([df_resampled, df_sample])
        
        return df_resampled
    
    
    def _clean_text(self, text):
        """
        Function to clean text.
        If a string is provided, it'll return a string. Otherwise, it'll return a Pandas Series.
        """
        
        if type(text) != pd.Series:
            cleaned_text = pd.Series(text)
        else:
            cleaned_text = text
        
        cleaned_text = cleaned_text.str.lower()
        cleaned_text = cleaned_text.apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
        
        if type(text) == str:
            cleaned_text = cleaned_text[0]
        return cleaned_text
    
    
def output_score(y_test, y_pred, labels, score_method):
    score = score_method(y_test, y_pred)
    print2.bold(f"Accuracy: {round(score*100, 2)}%")
    print2.bold("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels = labels)
    df = pd.DataFrame(cm, columns=[labels], index = [labels])
    print(df)


def print_samples(n, df, filter, title = None):
    if title is not None: print2.bold(title)
    no_samples = n
    
    sample_indexes = df[filter].index.unique()[0:no_samples]
    samples = df[~df.index.duplicated()].loc[sample_indexes]

    for row in samples:
        print(row)