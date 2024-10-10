import time

import util

import torch
from tqdm.notebook import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ====================== Set important parameters ======================
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = '/home/FYP/c200129/playground/test2/models/'

OBJ_MODEL = 'BERT_reddit_obj_relabel_epoch_20.model'
SENT_MODEL = 'BERT_reddit_sent_relabel_epoch_21.model'
SARCASM_MODEL = 'finetuned_BERT_sacarsm_50k_mixed_epoch_5.model'

ALL_TIME_TAKEN = {}
ALL_ACC = {}
ALL_F1 = {}

OBJ_KEY = 'Objectivity Detection'
SENT_KEY = 'Sentiment Analysis'
SARCASM_KEY = 'Sarcasm Detection'

# ====================== Load reddit dataset ======================

# To be updated with path to evaluationd ataset
df_all = pd.read_csv(
    '/home/FYP/c200129/playground/test2/datasets/new_sampled_posts.csv',
    usecols=['post_id', 'post_title', 'post_body', 'changed_labels_combined']
)

train = pd.read_csv(
    '/home/FYP/c200129/playground/test2/training_post_id_relabelled.csv',
    header=None,
    names=['post_id']
)

df = df_all[~df_all['post_id'].isin(train['post_id'])].drop(columns=['post_id']).reset_index(drop=True)

# ====================== Process reddit dataset ======================

df = util.process_evaluation_data(df)

df, obj_label_dict = util.process_labels(df, 'obj')

# Encode and load data
obj_input_ids, obj_attention_masks, obj_labels = util.encode_data(df, 'objectivity label')
obj_dataset = TensorDataset(obj_input_ids, obj_attention_masks, obj_labels)
dataloader_obj = DataLoader(
    obj_dataset, 
    sampler=RandomSampler(obj_dataset), 
    batch_size=BATCH_SIZE
)

# ====================== Objectivity Detection ======================

# Create model instance
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(obj_label_dict),
    output_attentions=False,
    output_hidden_states=False
).to(DEVICE)

# Load objectivity model
print('='*35, ' Objectivity Detection ', '='*35)

model.load_state_dict(torch.load(PATH+OBJ_MODEL, map_location=torch.device('cuda')))
print(f'\nModel {OBJ_MODEL} loaded, getting predictions...=')

# Test objectivity model
start = time.time()

_, predictions, true_vals = util.evaluate(model, DEVICE, dataloader_obj)

time_taken = time.time() - start
ALL_TIME_TAKEN[OBJ_KEY] = round(time_taken, 3)

# Get scores
acc_dict_obj, obj_preds_flat = util.accuracy_per_class(predictions, true_vals, obj_label_dict)
f1_score_obj = util.f1_score_func(predictions, true_vals)
print(f'F1 score: {f1_score_obj:.3f}')

ALL_ACC[OBJ_KEY] = acc_dict_obj
ALL_F1[OBJ_KEY] = round(f1_score_obj, 3)

cm = confusion_matrix(true_vals, obj_preds_flat)
print(f'Confusion matrix:\n{cm}\n')

# ====================== Sentiment Analysis ======================

# Prepare dataset
df_sent = df[obj_preds_flat == 1].drop(columns=['objectivity label'])
df_sent = df_sent[~df_sent['sentiment'].isnull()]
df_sent, sent_label_dict = util.process_labels(df_sent, 'sent')

sent_input_ids, sent_attention_masks, sent_labels = util.encode_data(df_sent, 'sentiment label')
sent_dataset = TensorDataset(sent_input_ids, sent_attention_masks, sent_labels)
dataloader_sent = DataLoader(
    sent_dataset, 
    sampler=SequentialSampler(sent_dataset), 
    batch_size=BATCH_SIZE
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(sent_label_dict),
    output_attentions=False,
    output_hidden_states=False
).to(DEVICE)

# Load sentiment analysis model
print('='*35, ' Sentiment Analysis ', '='*35)

model.load_state_dict(torch.load(PATH+SENT_MODEL, map_location=torch.device('cuda')))
print(f'Model {SENT_MODEL} loaded, getting predictions...')

# Test sentiment analysis model
start = time.time()
_, sent_predictions, sent_true_vals = util.evaluate(model, DEVICE, dataloader_sent)
time_taken = time.time() - start

ALL_TIME_TAKEN[SENT_KEY] = round(time_taken, 3)

# Get scores
acc_dict_sent, sent_preds_flat = util.accuracy_per_class(sent_predictions, sent_true_vals, sent_label_dict)
f1_score_sent = util.f1_score_func(predictions, true_vals)
print(f'F1 score: {f1_score_sent:.3f}')

ALL_ACC[SENT_KEY] = acc_dict_sent
ALL_F1[SENT_KEY] = round(f1_score_sent, 3)

cm = confusion_matrix(sent_true_vals, sent_preds_flat)
print(f'Confusion matrix:\n{cm}\n')

# ====================== Sarcasm Detection ======================

# Get list of wrongly classified sentences for sentiment analysis
df_wrong_sent = df_sent.drop(columns=['sentiment label', 'type'])
df_wrong_sent['predictions label'] = sent_preds_flat

# Prepare dataset
label_dict_inverse = {v: k for k, v in sent_label_dict.items()}
df_wrong_sent['prediction'] = df_wrong_sent['predictions label'].map(label_dict_inverse)
df_wrong_sent = df_wrong_sent[df_wrong_sent['sentiment'] != df_wrong_sent['prediction']]

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(sent_label_dict),
    output_attentions=False,
    output_hidden_states=False
).to(DEVICE)

print('='*35, ' Sarcasm Detection ', '='*35)

model.load_state_dict(torch.load(PATH+SARCASM_MODEL, map_location=DEVICE))
print(f'Model {SARCASM_MODEL} loaded')

# Test for sarcasm detection
start = time.time()
test_preds_flat = util.test(df_wrong_sent, model, DEVICE, BATCH_SIZE)
time_taken = time.time() - start

ALL_TIME_TAKEN[SARCASM_KEY] = time_taken

# Reverse prediction for wrongly classified rows
df_sarcasm = df_wrong_sent.copy()
df_sarcasm['sarcasm'] = test_preds_flat
df_sarcasm['new prediction'] = df_sarcasm.apply(util.apply_sarcasm, axis=1)
df_sarcasm = df_sarcasm[df_sarcasm['predictions label'] != df_sarcasm['new prediction']]

# Clean up columns
df_sarcasm = df_sarcasm.drop(columns=['prediction', 'sarcasm', 'predictions label'])
df_sarcasm = df_sarcasm.rename(columns={'new prediction': 'predictions label'})
df_sarcasm.head()

# Add prediction labels to sentiment dataframe
df_sent['predictions label'] = sent_preds_flat

# Update predicitons in wrongly classified data
df_final = df_sent.copy()
df_final.update(df_sarcasm['predictions label'])
final_preds_flat = np.array(df_final['predictions label'])

# Get new scores after sarcasm detection
acc_dict_sarcasm, sarcasm_preds_flat = util.accuracy_per_class(None, sent_true_vals, sent_label_dict, final_preds_flat)
f1_score_sarcasm = util.f1_score_func(None, sent_true_vals, final_preds_flat)
print(f'F1 score: {f1_score_sarcasm:.3f}')

ALL_ACC[SARCASM_KEY] = acc_dict_sarcasm
ALL_F1[SARCASM_KEY] = ALL_F1[SENT_KEY] = round(f1_score_sarcasm, 3)

cm = confusion_matrix(sent_true_vals, final_preds_flat)
print(f'Confusion matrix:\n{cm}\n')

# ====================== Get average scores ======================
print('='*35, ' Average Performance ', '='*35)


# Get average before sarcasm detection
keys = [OBJ_KEY, SENT_KEY]

avg_acc_bef_sarc = util.get_avg_acc(keys, ALL_ACC)
print(f'Average accuracy before sarcasm detection: {avg_acc_bef_sarc}')

# Get average after sarcasm detection
keys = [OBJ_KEY, SARCASM_KEY]

avg_acc_aft_sarc = util.get_avg_acc(keys, ALL_ACC)
print(f'Average accuracy after sarcasm detection: {avg_acc_aft_sarc}\n')

avg_f1_bef_sarc = (ALL_F1[OBJ_KEY] + ALL_F1[SENT_KEY])/2
avg_f1_aft_sarc = (ALL_F1[OBJ_KEY] + ALL_F1[SARCASM_KEY])/2

print(f'Average F1 score before sarcasm detection: {avg_f1_bef_sarc:.3f}')
print(f'Average F1 score after sarcasm detection: {avg_f1_aft_sarc:.3f}')
print()

print('='*35, ' All Stats ', '='*35)

print(ALL_TIME_TAKEN)
print()
print(ALL_ACC)
print()
print(ALL_F1)
