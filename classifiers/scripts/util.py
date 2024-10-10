import re
import nltk
from nltk import word_tokenize
import string
import emoji

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm.notebook import tqdm
from transformers import BertTokenizer

import time

def remove_not_ASCII(text):
    """
    Remove non-ASCII characters

    Args:
        text (str): text to be processed

    Returns:
        text: processed text
    """
    text = str(text)
    text = ''.join([word for word in text if word in string.printable])
    return text

def replace_emoticons(text):
    """
    Replace emoticons such as :) and :( with their text equivalent

    Args:
        text (str): text to be processed

    Returns:
        text: processed text
    """
    text = text.replace("<3", "heart ")
    text = re.sub('>:-?\'?"?\(+', 'angry ', text)
    text = re.sub('\)+:-?\'?"?:<', 'angry ', text)
    text = re.sub(':-?\'?"?(o+|O+|0+)', 'surprised ', text)
    text = re.sub(':-?\'?"?(\)+|>+|D+)', 'smile ', text)
    text = re.sub('(\(+|<+)-?\'?"?:', 'smile ', text)
    text = re.sub(':-?\'?"?\(+', 'sad ', text)
    text = re.sub('(\)+|>+|D+)-?\'?"?:', 'sad ', text)
    
    return text

def text_preprocessing(text):
    """
    Clean the text by removing unwanted things and convert emojis into text

    Args:
        text (str): text to be processed

    Returns:
        text: processed text
    """

    text = replace_emoticons(text)                           # convert emoticon to text
    text = emoji.demojize(text, delimiters=("", " "))        # convert emoji to text
    text = remove_not_ASCII(text)                            # remove non-ASCII characters

    text = re.sub('\s+', ' ', text)
    text = re.sub('<br />', '', text)                        # remove <br />
    text = re.sub('\(?https?:\/\/\S+', '', text)             # remove URLs
    text = re.sub('&.+;', '', text)                          # remove unknown unicode characters
    text = re.sub('\\n', '', text)                           # remove \n literals
    
    text = re.sub('u/\S+', 'user', text)                     # replace user mentions
    text = re.sub('@\S+', 'user', text)
    text = re.sub('r/\S+', 'subreddit', text)                # replace subreddit mentions
    return text


def pos_tag(text):
    """
    Get POS tags

    Args:
        text (str): text to get POS tags

    Returns:
        tags: list of (word, tag) tuples
    """
    # Tokenise and get POS tags of each word
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    return tags

def embed_pos(arr):
    """
    Convert POS tags to embeddings by concatenating them with the words

    Args:
        arr (list): list of (word, tag) tuples

    Returns:
        new_arr: list of processed tokens in the form of word_tag
    """
    new_arr = ""
    for token in arr:
        combined = f'{token[0]}_{token[1]}'
        new_arr += combined + ' '
    return new_arr

def process_evaluation_data(df):
    """
    Preprocess raw data, including joining post title with post body, creating a separate 'sentiment'
    column and applying text preprocessing.

    Args:
        df (dataframe): dataframe to be processed

    Returns:
        df: processed dataframe
    """

    # Combine post title and post body
    df['sentence'] = df['post_title'] + df['post_body'].fillna('')
    df.drop(columns=['post_title', 'post_body'], inplace=True)
    df = df.rename(columns={"changed_labels_combined": "type"})

    # Separate values in type column
    df['sentiment'] = df['type'][df['type'] != 'Objective']
    df['type'] = df['type'].replace(['Positive', 'Negative'], 'Subjective')
    df = df[['sentence', 'type', 'sentiment']]

    df['sentence'] = df['sentence'].apply(text_preprocessing)

    return df

def process_labels(df, subtask):
    """
    Create labels that the model can understand.

    Args:
        df (dataframe): input data
        subtask (str): whether it is objectivity detection or sentiment analysis

    Returns:
        df: altered dataframe
        label_dict: dictionary of possible labels
    """
    if subtask == 'obj':
        label_name = 'objectivity label'
        label_col = 'type'
    else:
        label_name = 'sentiment label'
        label_col = 'sentiment'

    possible_labels = df[label_col].unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df[label_name] = df[label_col].replace(label_dict)

    return df, label_dict

def encode_data(df, df_label):
    """
    Args:
        df: dataframe containing text data and class labels, text data column must be named 'sentence'
        df_label (str): column name of the class labels
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', 
        do_lower_case=True
    )

    encoded_data = tokenizer.batch_encode_plus(
        df.sentence.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    labels = torch.tensor(df[df_label].values)

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']

    return input_ids, attention_masks, labels


# Using Early stopper to stop when the F1 Score prediction drops 
class EarlyStopper:
    """
    Checks for early stopping when the selected scoring metric decreased by a number of times consecutive,
    defined by patience value
    """
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_f1_score = 0
        self.min_accuracy = 0

    def early_stop_f1(self, f1_score):
        """
        Early stop using F1 score

        Args:
            f1_score (float): current F1 score obtained

        Returns:
            bool: True if early stopping, else False
        """
        if f1_score > self.min_f1_score:
            self.min_f1_score = f1_score
            self.counter = 0
        elif f1_score < (self.min_f1_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                self.min_f1_score = 0
                return True
        return False

    def early_stop_accuracy(self, accuracy):
        """
        Early stop using accuracy

        Args:
            f1_score (float): current accuracy obtained

        Returns:
            bool: True if early stopping, else False
        """
        if accuracy > self.min_accuracy:
            self.min_accuracy = accuracy
            self.counter = 0
        elif accuracy < (self.min_accuracy + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                self.min_accuracy = 0
                return True
        return False


def f1_score_func(preds, labels, preds_flat=None):
    """
    Returns F1 score calculated by sklearn's f1_score function
    
    Args:
        preds: prediction by model
        labels: true labels
        
    Returns:
        f1_score: calculated F1 score
    
    """
    if preds_flat is None:
        preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels, label_dict, preds_flat=None):
    """
    Calculates the accuracy values based on each class and overall accuracy

    Args:
        preds (np.array): array of unflattened predictions
        labels (np.array): array of true labels
        label_dict (dict): dictionary of labels and their respective values
        preds_flat (np.array, optional): array of flattened predictions. Defaults to None.

    Returns:
        dict: dictionary of accuracy values for each class and overall
        np.array: array of flattened predictions
    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    if preds_flat is None:
        preds_flat = np.argmax(preds, axis=1).flatten()
        
    labels_flat = labels.flatten()
    
    total_preds = 0
    total_correct = 0
    true_dict, acc_dict, pred_dict = {}, {}, {}

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
                
        total_preds += len(y_preds)
        num_correct = len(y_preds[y_preds==label])
        total_correct += num_correct
        acc = num_correct/len(y_true)
        
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {acc*100:.3f}%, {num_correct}/{len(y_true)}')
        
        true_dict[label] = len(y_true)
        pred_dict[label] = num_correct
        acc_dict[label] = round(acc*100, 3)
    
    total_acc = (total_correct/total_preds)*100
    print(f'Total accuracy: {total_acc:.3f}%')
    print('-'*50, '\n')

    acc_dict = {
        "class distribution": true_dict,
        "correct predictions per class": pred_dict,
        "accuracy per class": acc_dict,
        "total accuracy": round(total_acc, 3)
    }

    return acc_dict, preds_flat

def train(model, model_name, epochs, device, dataloader_train, dataloader_val, optimizer, scheduler):
    """
    Trains a BERT base instance

    Args:
        model (BERT base): BERT base model instance
        model_name (str): name of the file to store the model
        epochs (int): max number of epochs to train on
        device (torch.device): 'cuda' if GPU is available, else 'cpu'
        dataloader_train (dataloader): dataloader object for the training data
        dataloader_val (dataloader): dataloader object for the validation data
        optimizer (optimizer): model optimizer
        scheduler (scheduler): model scheduler

    Returns:
        time: time taken to train the model
        e: number of epochs used for training
    """
    early_stopper = EarlyStopper(patience=3, min_delta=0)
    start = time.time()

    for epoch in tqdm(range(1, epochs+1)):

        start_epoch = time.time()
        
        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:
            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
            
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(model, device, dataloader_val)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

        time_taken_epoch = time.time() - start_epoch
        print(f'Time taken to train epoch {epoch}: {time_taken_epoch:.3f} seconds')
        
        # Check for early stop
        if early_stopper and early_stopper.early_stop_f1(val_f1):
            e = epoch
            print(f"Early stopping at epoch {epoch} due to no improvement in F1 score.")
            torch.save(model.state_dict(), f'{model_name}_{e}.model')
            break
    

    time_taken = time.time() - start
    print(f'time taken to train: {time_taken:.3f} seconds')
    print('='*50)
    print()

    return time_taken, e

def evaluate(model, device, dataloader_val):
    """
    Evaluate the model on the validation dataset

    Args:
        model (BERT): BERT instance to be validated
        device (torch.device): 'cuda' if GPU is available, else 'cpu'
        dataloader_val (dataloader): dataloader object for the validation dataset

    Returns:
        loss_val_avg: float, average loss during validation
        predictions: numpy array, list of predictions from model
        true_vals: numpy array, list of true labels
    """

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def test(df, model, device, batch_size):
    """
    Test the model on unseen data

    Args:
        df (dataframe): data to be tested on
        model (BERT): BERT model instance
        device (torch.device): 'cuda' if GPU is available, else 'cpu'
        batch_size (int): size of a batch

    Returns:
        test_preds_flat: numpy array of flattened predictions
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', 
        do_lower_case=True
    )

    # Encode data
    encoded_text = tokenizer.batch_encode_plus(
        df.sentence.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )

    input_ids_text = encoded_text['input_ids']
    attention_masks_text = encoded_text['attention_mask']

    # Create dataloader
    dataset = TensorDataset(input_ids_text, attention_masks_text)
    dataloader = DataLoader(
        dataset, 
        sampler=RandomSampler(dataset), 
        batch_size=batch_size
    )
    
    # Test model
    model.eval()
    test_predictions = []

    for batch in dataloader:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        test_predictions.append(logits)

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_preds_flat = np.argmax(test_predictions, axis=1).flatten()

    return test_preds_flat
    
def apply_sarcasm(row):
    """
    Reverse the sentiment label of a sarcasm remark

    Args:
        row (dataframe): a row in a dataframe

    Returns:
        int: the new label
    """
    if row['sarcasm'] == 0:
        return row['predictions label']
    elif row['sarcasm'] == 1:
        if row['predictions label'] == 1:
            return 0
        else:
            return 1

def get_avg_acc(keys, acc_dict):
    """
    Calculate the overall accuracy of the entire project

    Args:
        keys (list): list of keys to be used to retrieve the required stats
        acc_dict (dict): accuracy dictionary containing the required stats

    Returns:
        avg_acc: float, average accuracy calculated
    """
    total_entries = 0
    total_correct = 0

    for key in keys:
        for _, value in acc_dict[key]['class distribution'].items():
            total_entries += value
        for _, value in acc_dict[key]['correct predictions per class'].items():
            total_correct += value

    avg_acc = round(total_correct/total_entries, 3)
    return avg_acc       

def get_avg_f1(obj_true_vals, obj_preds_flat, sent_true_vals, sent_preds_flat, sarc_preds_flat):
    obj_precision = precision_score(obj_true_vals, obj_preds_flat)
    obj_recall = recall_score(obj_true_vals, obj_preds_flat)

    sent_precision = precision_score(sent_true_vals, sent_preds_flat)
    sent_recall = recall_score(sent_true_vals, sent_preds_flat)

    macro_precision = (obj_precision + sent_precision)/2
    macro_recall = (obj_recall + sent_recall)/2
    average_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    print(f'Average F1 score before sarcasm detection: {average_f1:.3f}')
    
    sarc_precision = precision_score(sent_true_vals, sarc_preds_flat)
    sarc_recall = recall_score(sent_true_vals, sarc_preds_flat)

    macro_precision = (obj_precision + sarc_precision)/2
    macro_recall = (obj_recall + sarc_recall)/2
    average_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    print(f'Average F1 score after sarcasm detection: {average_f1:.3f}')
