import os
import sys

import requests
import openai
import mlflow.keras
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from ast import literal_eval
from wordcloud import WordCloud  
from collections import Counter
from collections import defaultdict
from sklearn.cluster import KMeans
from openai.embeddings_utils import get_embedding, cosine_similarity
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

class IResult:
    def get_result(self, df):
        pass  


class Embedding(IResult):
    def get_result(self, df):
        df['ada_v2'] = df["context"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002'))
        return df


class Key_Phrase(IResult):
    def get_result(self, df): 
        result = []
        batch_size = 10
        label_list = list(df['context'])

        for i in range(0, len(label_list), batch_size):
            batch = label_list[i:i+batch_size]
            response = text_analytics_client.extract_key_phrases(batch, language="zh-hant")

            for idx, doc in enumerate(response):
                key_list = doc.key_phrases
                result.append(key_list)
        df['key_phrase'] = [result[i] for i in range(len(result))]
        return df 

class Auto_Encoder(IResult):
    def get_result(self, df): 
        model = mlflow.keras.load_model("{run ID}")
        model.trainable = False
        train_df = np.stack(df['ada_v2'].values)
        bottle_neck = model.model_encoder(train_df).numpy()
        df['bottle_neck'] = [bottle_neck[i] for i in range(len(bottle_neck))]
        return df
    
class K_Means(IResult):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def get_result(self, df):
        train_df = np.stack(df['bottle_neck'].values)
        estimator = KMeans(n_clusters=self.n_clusters, random_state=24)  
        estimator.fit(np.stack(df['bottle_neck'].values))
        labels = estimator.labels_
        df['label'] = labels
        return df

class Pipeline:
    def __init__(self, process_list,df):
        self.process_list = process_list
        self.df = df

    def run(self):
        for obj in self.process_list:
            try:
                print(obj.__class__)
                self.df = obj.get_result(self.df)
            except Exception as e:
                print(obj.__class__,e)
        return self.df

def main(df):
    process_list = [Embedding(), Key_Phrase(),Auto_Encoder(), K_Means(4)] 
    pipeline = Pipeline(process_list, df)
    result_df = pipeline.run()