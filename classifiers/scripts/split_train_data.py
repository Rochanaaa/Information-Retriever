import util

import pandas as pd
from sklearn.model_selection import train_test_split

# Change this parameter to toggle between objectivity and sentiment analysis 
TRAIN_TYPE = 'sentiment'

# Load reddit dataset
df = pd.read_csv('/home/FYP/c200129/playground/test2/datasets/new_sampled_posts.csv',
                   usecols=['post_title', 'post_body', 'changed_labels_combined'])

df['sentence'] = df['post_title'] + ' ' + df['post_body'].fillna('')
df.drop(columns=['post_title', 'post_body'], inplace=True)
df = df.rename(columns={"changed_labels_combined": "type"})



df['sentence'] = df['sentence'].apply(util.text_preprocessing)

# Get objectivity label
if TRAIN_TYPE == 'objectivity':
    file_name = f'_objectivity.csv'
    label_name = 'objectivity label'

    possible_labels = df.type.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    
    df[label_name] = df.type.replace(label_dict)

else:
    file_name = f'_sentiment.csv'
    label_name = 'sentiment label'

    df = df.dropna(subset=['sentiment']).reset_index(drop=True)
    df = df.drop(columns=['type'])

    possible_labels = df.sentiment.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df[label_name] = df.sentiment.replace(label_dict)

X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df[label_name].values, 
                                                  test_size=0.7, 
                                                  random_state=11, 
                                                  stratify=df[label_name].values)

df_train = df[df.index.isin(X_train)]
df_train.to_csv(f'train{file_name}', index=False)

df_val = df[~df.index.isin(X_train)]
df_val.to_csv(f'val_{file_name}', index=False)
