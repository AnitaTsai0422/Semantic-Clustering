import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from ast import literal_eval
from wordcloud import WordCloud  
from collections import Counter
from collections import defaultdict

class TF_IDF:
    def __init__(self, df):
        self.df = df

    def _get_tf(self):
        df_copy = self.df.copy()
        result_list = []

        k = df_copy['label'].nunique()
        for i in range(k):
            data_label = [''.join(item) for sublist in df_copy[df_copy['label'] == i]['key_phrase'] for item in sublist]
            data_label_dic = Counter(data_label)
            df_data_label = pd.DataFrame({'term': data_label_dic.keys(), 
                                        'tf': [v / sum(data_label_dic.values()) for v in data_label_dic.values()], 
                                        'label': i})
            result_list.append(df_data_label)

        result = pd.concat(result_list, ignore_index=True)
        return result

    def _get_idf(self):
        df_copy = self.df.copy()
        k = df_copy['label'].nunique()
        total_key_phrase = [''.join(item) for sublist in df_copy['key_phrase'] for item in sublist]
        unique_list = list(set(total_key_phrase))
        label_lists = [set([''.join(item) for sublist in df_copy[df_copy['label'] == i]['key_phrase'] for item in sublist]) for i in range(k)]
        result = {element: sum(element in lst for lst in label_lists) for element in unique_list}
        data = pd.DataFrame([{'term': term, 'idf': math.log(round( k / count, 2))} for term, count in result.items()])
        return data

    def get_tf_idf(self):
        data_tf = self._get_tf()
        data_idf = self._get_idf()

        output_dfs = []
        labels = data_tf['label'].unique()

        for label in labels:
            df_label = data_tf[data_tf['label'] == label]
            df = df_label.merge(data_idf, how='left', on='term')
            df['tf-idf'] = df['tf'] * df['idf']
            output_dfs.append(df)

        result = pd.concat(output_dfs, ignore_index=True)
        return result
    
def _label_to_dic(final_data):
    k = final_data['label'].nunique()
    result = {}
    for i in range(k):
        dd = defaultdict(list)
        label_list = final_data[final_data['label'] == i].to_dict('records', into=dd)
        result[i] = {item['term']: item['tf-idf'] for item in label_list}
    return result

def plot_word_cloud(dic):
    TC_FONT_PATH = '{path}'
    k = len(dic.keys())
    # k = data_final['label'].nunique()
    # dic = _label_to_dic()

    for i in range(k):
        wordcloud = WordCloud(width = 1000, height = 500,font_path=TC_FONT_PATH).generate_from_frequencies(dic[i]) 
        plt.figure(figsize=(15,8))
        plt.imshow(wordcloud)

def main(df):
    tf_idf = TF_IDF(result_df)
    test_df = tf_idf.get_tf_idf()
    result = _label_to_dic(test_df)
    plot_word_cloud(result)

if __name__ == '__main__':
    main()