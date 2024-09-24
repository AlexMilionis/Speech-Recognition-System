#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import os
from glob import glob

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import math

# BHMA 9
def parse_free_digits(directory):
    
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [os.path.basename(f).split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)
    

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)
        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(y=wav, sr=Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc).T
        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))
    return frames


def split_free_digits(frames, ids, speakers, labels ):
    print("Splitting in train dev test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_dev, y_dev, spk_dev = [], [], []
    X_test, y_test, spk_test = [], [], []
    
    test_indices = ["0", "1", "2", "3", "4"]
    dev_indices  = ["5","6","7","8","9","10","11","12","13"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        elif str(idx) in dev_indices:
            X_dev.append(frame)
            y_dev.append(label)
            spk_dev.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)
            

    return X_train, X_dev, X_test, y_train, y_dev, y_test, spk_train, spk_dev, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))

    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled

    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_dev, X_test, y_train, y_dev, y_test, spk_train, spk_dev, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_dev, X_test, y_train, y_dev, y_test, spk_train, spk_dev, spk_test

directory = 'free-spoken-digit-dataset/recordings/'
X_train, X_dev, X_test, y_train, y_dev, y_test, spk_train, spk_dev, spk_test = parser(directory)

print("If using X_train + X_dev to calculate normalization statistics")
scale_fn = make_scale_fn(X_train + X_dev)
X_train = scale_fn(X_train)
X_dev = scale_fn(X_dev)
X_test = scale_fn(X_test)

# BHMA 10

# Gather data separately for each digit
def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic


def create_data():
    
    X_train, X_val, X_test, y_train, y_val, y_test, spk_train, spk_val, spk_test = parser(directory, n_mfcc=13)

    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels

def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        d = Normal().fit(np.concatenate(X))
        dists.append(d)
    return dists

def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):

    dists = []
    for i in range(n_states):
        distributions = [ Normal() ] * n_mixtures
        a = GeneralMixtureModel(distributions, verbose=True).fit( np.concatenate(X).astype('float32') )
        dists.append(a)
    return dists

def initialize_transition_matrix(n_states):
    A = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states - 1):
        A[i, i] = 0.5      # Probability of staying in the same state
        A[i, i + 1] = 0.5  # Probability of transitioning to the next state
    A[n_states - 1, n_states - 1] = 1.0  # The last state is absorbing
    return A

def initialize_starting_probabilities(n_states):
    start_probs = np.zeros(n_states, dtype=np.float32)
    start_probs[0] = 1.0  # Start in the first state
    return start_probs


def initialize_end_probabilities(n_states):
    end_probs = np.zeros(n_states, dtype=np.float32)
    end_probs[n_states - 1] = 1.0  # Can only end in the last state
    return end_probs

def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [x.astype(np.float32) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
    ).fit(data)
    return model


def train_hmms(train_dic, labels):
    hmms = {}  # create one hmm for each digit
    for dig in labels:
        X, _, _, _ = train_dic[dig]
        print(f'digit = {dig}')
        if gmm==True:
            emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
        else:
            emission_model = initialize_and_fit_normal_distributions(X, n_states)
        hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = [0] * len(labels)
            sample = np.expand_dims(sample, 0)
            for digit, hmm in hmms.items():
                logp = hmm.log_probability(sample)  # use the hmm.log_probability function
                ev[digit] = logp

            predicted_digit = ev.index(max(ev))  # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true


train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()


n_states = 3
n_mixtures = 2
gmm = True  
covariance_type = "diag"  

hmms = train_hmms(train_dic, labels)
labels = list(set(y_train))
pred_val, true_val = evaluate(hmms, val_dic, labels)
pred_test, true_test = evaluate(hmms, test_dic, labels)

# Calculate and print the accuracy score on the validation and the test sets
accuracy_val = accuracy_score(true_val, pred_val)
accuracy_test = accuracy_score(true_test, pred_test)
print(f'Validation Accuracy: {accuracy_val}')
print(f'Test Accuracy: {accuracy_test}')

# For Validation Set
cm = confusion_matrix(true_val, pred_val)
plot_confusion_matrix(cm, classes=labels, title='Confusion Matrix for Validation Set')
plt.show()

# For Test Set
cm = confusion_matrix(true_test, pred_test)
plot_confusion_matrix(cm, classes=labels, title='Confusion Matrix for Test Set')
plt.show()

## BHMA 14

output_dim = 10  # number of digits
rnn_size = 64
num_layers = 2
bidirectional = True 
dropout =  0.4         
batch_size = 32
patience = 3
epochs = 100
lr = 1e-4
weight_decay = 1e-5 

class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode

    def stop(self, value: float) -> bool:
        # Decrease patience if the metric hs not improved
        # Stop when patience reaches zero
        if self.has_improved(value):
            self.best = value
            self.patience_left = self.patience
        else:
            self.patience_left -= 1
        return self.patience_left <= 0
        

    def has_improved(self, value: float) -> bool:
        # Check if the metric has improved
        if self.mode == "min":
            return self.best is None or value < self.best
        elif self.mode == "max":
            return self.best is None or value > self.best
        else:
            raise ValueError("Mode must be 'min' or 'max'")


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: Python list of numpy arrays that contain the sequence features.
               Each element of this list is a numpy array of shape seq_length x feature_dimension
        labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths = [len(i) for i in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        """
        This function performs zero padding on a list of features and 
        forms them into a numpy 3D array
        returns
            padded: a 3D numpy array of shape num_sequences x 
            max_sequence_length x feature_dimension
        """
        max_length = max(map(len, x)) # calculate max length
        padded_arr_list = []
        for arr in x:
            padding_length = max_length - len(arr) 
            padding = np.zeros((padding_length, arr.shape[1]))
            padded_arr = np.vstack((arr, padding))
            padded_arr_list.append(padded_arr)
            
        padded = np.stack(padded_arr_list) # create array from list
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, 
                 bidirectional=False, dropout=0.0,):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=rnn_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if 
                            num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """

        # recognizes the zeros at the end of the sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, 
                                                     batch_first=True, enforce_sorted=False)
        # forward pass for packed sequence
        packed_output, (hidden, cell) = self.lstm(x_packed)
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    
        last_outputs: torch.Tensor = self.last_timestep(output, lengths, self.bidirectional)
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()


def create_dataloaders(batch_size):
    X_train, X_val, X_test, y_train, y_val, y_test, spk_train, spk_val, spk_test = parser(
        directory, n_mfcc=13)

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)
    # Initialize the training, val and test dataloaders (torch.utils.data.DataLoader)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        features = features.float()
        # zero grads in the optimizer
        optimizer.zero_grad()
        # run forward pass
        outputs = model(features, lengths)
        # calculate loss
        loss = criterion(outputs, labels)
        # Run backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch
            features = features.float()
            # Run forward pass
            logits = model(features, lengths)
            # calculate loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predict
            outputs = torch.argmax(logits, dim=1)  # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs))
            y_true = torch.cat((y_true, labels))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(train_dataloader, val_dataloader, criterion):
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # Initialize AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience, mode="min")
    train_losses = []
    val_losses   = []
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

        # Calculate and print accuracy score
        valid_accuracy = (y_pred == y_true).float().mean()
        print(
            "Epoch {}: train loss = {}, valid loss = {}, valid acc = {}".format(
                epoch, training_loss, valid_loss, valid_accuracy
            )
        )
        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break
        train_losses.append(training_loss)
        val_losses.append(valid_loss)
    return model,train_losses,val_losses


train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)
# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()
model, train_losses, val_losses = train(train_dataloader, val_dataloader, criterion)

test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)

test_accuracy = (test_pred == test_true).float().mean()
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

cm = confusion_matrix(test_pred, test_true)
plot_confusion_matrix(cm, classes=labels, title='Confusion Matrix for Test Set')
plt.show()
