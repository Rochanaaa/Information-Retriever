import util

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Change this parameter to toggle between objectivity and sentiment analysis 
TRAIN_TYPE = 'sentiment'

# Load reddit dataset
df = pd.read_csv('/home/FYP/c200129/playground/test2/datasets/new_sampled_posts.csv',
                   usecols=['post_title', 'post_body', 'changed_labels_combined'])

df['sentence'] = df['post_title'] + ' ' + df['post_body'].fillna('')
df.drop(columns=['post_title', 'post_body'], inplace=True)
df = df.rename(columns={"changed_labels_combined": "type"})

# Separate values in type column
df['sentiment'] = df['type'][df['type'] != 'Objective']
df['type'] = df['type'].replace(['Positive', 'Negative'], 'Subjective')

df['sentence'] = df['sentence'].apply(util.text_preprocessing)

# Get objectivity label
if TRAIN_TYPE == 'objectivity':
    model_name = f'BERT_reddit_obj_relabel_epoch'
    label_name = 'objectivity label'

    possible_labels = df.type.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    
    df[label_name] = df.type.replace(label_dict)

else:
    model_name = f'BERT_reddit_sent_relabel_epoch'
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

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

input_ids_train, attention_masks_train, labels_train = util.encode_data(df,label_name, 'train')
input_ids_val, attention_masks_val, labels_val = util.encode_data(df, label_name, 'val')

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


batch_size = 32
num_workers = 0

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size,
                              num_workers=num_workers)

dataloader_val = DataLoader(dataset_val, 
                            sampler=SequentialSampler(dataset_val), 
                            batch_size=batch_size,
                            num_workers=num_workers)                                                 


optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

epochs = 50

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


seed_val = 11
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

time_taken, e = util.train(model, model_name, epochs, device, dataloader_train, dataloader_val, optimizer, scheduler)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load(f'{model_name}_{e}.model', map_location=device))
print(f'Testing {model_name}_{e}.model... ')

_, predictions, true_vals = util.evaluate(model, device, dataloader_val)

_, preds_flat = util.accuracy_per_class(predictions, true_vals, label_dict)
print(f"F1 score: {util.f1_score_func(predictions, true_vals)}")

cm = confusion_matrix(true_vals, preds_flat)
print(cm)