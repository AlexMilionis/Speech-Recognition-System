# # Import necessary libraries
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn

############################################################################################################
## BHMA 2
FOLDER_PATH = 'digits/'
def data_parser(folder):
    wav_list       = []
    speaker_list   = []
    digit_list     = []

    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            file_parts = filename.split('.')[0]  # Remove .wav
            speaker    = ''.join(filter(str.isdigit, file_parts))  # Extract number (1)
            digit      = ''.join(filter(str.isalpha, file_parts))  # Extract name (eight)

            file_path      = os.path.join(folder, filename)
            audio_data, fs = librosa.load(file_path, sr=None)

            wav_list.append(audio_data)
            speaker_list.append(speaker)
            digit_list.append(digit)

    return wav_list, speaker_list, digit_list, fs

wav_data, speakers, digits, fs = data_parser(FOLDER_PATH)

############################################################################################################
## BHMA 3
window = 25 * 10**(-3)  # Step size: 10ms
step   = 10 * 10**(-3)  # Step size: 10ms
sampling_rate = fs  # Hz
mfccs, mfcc_deltas, mfcc_delta_deltas = [],[],[]
window_in_samples = int(window*sampling_rate)
step_in_samples   = int(step*sampling_rate)
num_samples       = len(wav_data)

for i in range(num_samples):
    # Extract 13 MFCCs (133 wav files x 13 MFCCs x Frames/wav)
    mfccs.append(librosa.feature.mfcc(y=wav_data[i], sr=sampling_rate, n_mfcc=13, n_fft=window_in_samples,
                                  hop_length=step_in_samples) )

    # 133 wav x 13 delta_MFCCs x 75 Frames/wav
    mfcc_deltas.append( librosa.feature.delta(mfccs[i]) )
    # 133 wav x 13 delta2_MFCCs x 75 Frames/wav
    mfcc_delta_deltas.append( librosa.feature.delta(mfccs[i], order=2) )

############################################################################################################
# BHMA 4

n1 = 'six'   # Alexis Milionis: 03400226
n2 = 'seven' # Panagiotis Chronopoulos: 03400240

# keep indices of .wav with digits six or seven
n1_idx, n2_idx = [],[]
for i in range(num_samples):
    if digits[i]==n1:
        n1_idx.append(i)
    elif digits[i]==n2:
        n2_idx.append(i)

# keep MFCC1,MFCC2 of .wav files with digits six or seven
mfcc1_n1, mfcc2_n1, mfcc1_n2, mfcc2_n2 = [],[],[],[]

for i in range(num_samples):
    if i in n1_idx:
        mfcc1_n1.append( mfccs[i][0] ) # #files with digit n1 x 1 (MFCC=1) x #frames
        mfcc2_n1.append( mfccs[i][1] ) # #files with digit n1 x 1 (MFCC=2) x #frames
    if i in n2_idx:
        mfcc1_n2.append( mfccs[i][0] ) # #files with digit n2 x 1 (MFCC=1) x #frames
        mfcc2_n2.append( mfccs[i][1] ) # #files with digit n2 x 1 (MFCC=2) x #frames

#Flatten lists
# mfcc1
# Concatenate the arrays and then convert to a list
mfcc1_n1_flattened = np.concatenate(mfcc1_n1).tolist()
mfcc2_n1_flattened = np.concatenate(mfcc2_n1).tolist()
mfcc1_n2_flattened = np.concatenate(mfcc1_n2).tolist()
mfcc2_n2_flattened = np.concatenate(mfcc2_n2).tolist()

# Plot histograms
plt.figure(figsize=(14, 6))

# Histogram for the 1st MFCC of digit '0' and '1'
plt.subplot(2, 2, 1)
plt.hist(mfcc1_n1_flattened, bins=30, alpha=0.5, label=f"Digit {n1}")
plt.hist(mfcc1_n2_flattened, bins=30, alpha=0.5, label=f"Digit {n2}")
plt.title(f'Histogram of the 1st MFCC for digits {n1} and {n2}')
plt.legend()

# Histogram for the 2nd MFCC of digit '0' and '1'
plt.subplot(2, 2, 2)
plt.hist(mfcc2_n1_flattened, bins=30, alpha=0.5, label=f"Digit {n1}")
plt.hist(mfcc2_n2_flattened, bins=30, alpha=0.5, label=f"Digit {n2}")
plt.title(f'Histogram of the 2nd MFCC for digits {n1} and {n2}')
plt.legend()

plt.tight_layout()
plt.show()

# find the ids of speakers of digits n1, n2
n1_speakers = [speakers[i] for i in n1_idx]  # all speakers of n1
n2_speakers = [speakers[i] for i in n2_idx]  # all speakers of n2
# indices of the first two speakers of n1,n2
speakers_idx=[]                              # (index_user1_n1, index_user1_n2, index_user2_n1, index_user2_n2 )
for item in n1_speakers:
    if item in n2_speakers:
        speakers_idx.append(n1_speakers.index(item))
        speakers_idx.append(n2_speakers.index(item))
    if len(speakers_idx)==4:
        break

# we find the indices of .wav files using the speakers' indices
idx_user1_n1 = n1_idx[speakers_idx[0]]
idx_user1_n2 = n2_idx[speakers_idx[1]]
idx_user2_n1 = n1_idx[speakers_idx[2]]
idx_user2_n2 = n2_idx[speakers_idx[3]]
wavdata_user1_n1 = wav_data[idx_user1_n1]
wavdata_user1_n2 = wav_data[idx_user1_n2]
wavdata_user2_n1 = wav_data[idx_user2_n1]
wavdata_user2_n2 = wav_data[idx_user2_n2]

# melspectrogram MFSCs = #bands x #frames
melspectrogram_user1_n1 = librosa.feature.melspectrogram(y=wavdata_user1_n1, sr=sampling_rate, n_fft=window_in_samples, hop_length=step_in_samples, n_mels=13)
melspectrogram_user1_n2 = librosa.feature.melspectrogram(y=wavdata_user1_n2, sr=sampling_rate, n_fft=window_in_samples, hop_length=step_in_samples, n_mels=13)
melspectrogram_user2_n1 = librosa.feature.melspectrogram(y=wavdata_user2_n1, sr=sampling_rate, n_fft=window_in_samples, hop_length=step_in_samples, n_mels=13)
melspectrogram_user2_n2 = librosa.feature.melspectrogram(y=wavdata_user2_n2, sr=sampling_rate, n_fft=window_in_samples, hop_length=step_in_samples, n_mels=13)

# correlation matrices for chosen 4 .wav files, between the 13 MSFC's
corr_matrix_user1_n1 = np.corrcoef(melspectrogram_user1_n1)
corr_matrix_user1_n2 = np.corrcoef(melspectrogram_user1_n2)
corr_matrix_user2_n1 = np.corrcoef(melspectrogram_user2_n1)
corr_matrix_user2_n2 = np.corrcoef(melspectrogram_user2_n2)

## Plot correlations
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot correlation matrix for wav file user1_n1
sns.heatmap(corr_matrix_user1_n1, annot=False, cmap='coolwarm', square=True, ax=axes[0, 0])
axes[0, 0].set_title(f'Correlation Matrix of MFSCs for speaker: {n1_speakers[speakers_idx[0]]} and digit: {n1}')

# Plot correlation matrix for wav file user1_n2
sns.heatmap(corr_matrix_user1_n2, annot=False, cmap='coolwarm', square=True, ax=axes[0, 1])
axes[0, 1].set_title(f'Correlation Matrix of MFSCs for speaker: {n2_speakers[speakers_idx[1]]} and digit: {n2}')

# Plot correlation matrix for wav file user2_n1
sns.heatmap(corr_matrix_user2_n1, annot=False, cmap='coolwarm', square=True, ax=axes[1, 0])
axes[1, 0].set_title(f'Correlation Matrix of MFSCs for speaker: {n1_speakers[speakers_idx[2]]} and digit: {n1}')

# Plot correlation matrix for wav file user2_n2
sns.heatmap(corr_matrix_user2_n2, annot=False, cmap='coolwarm', square=True, ax=axes[1, 1])
axes[1, 1].set_title(f'Correlation Matrix of MFSCs for speaker: {n2_speakers[speakers_idx[3]]} and digit: {n2}')

plt.tight_layout()
plt.show()


# We find the MFCCs for the two users and digits n1,n2
mfccs_user1_n1 = mfccs[idx_user1_n1]
mfccs_user1_n2 = mfccs[idx_user1_n2]
mfccs_user2_n1 = mfccs[idx_user2_n1]
mfccs_user2_n2 = mfccs[idx_user2_n2]

# correlation matrices for all 4  chosen .wav files, between the 13 MFCCs
corr_matrix_user1_n1 = np.corrcoef(mfccs_user1_n1)
corr_matrix_user1_n2 = np.corrcoef(mfccs_user1_n2)
corr_matrix_user2_n1 = np.corrcoef(mfccs_user2_n1)
corr_matrix_user2_n2 = np.corrcoef(mfccs_user2_n2)

## Plot correlations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot correlation matrix for wav file user1_n1
sns.heatmap(corr_matrix_user1_n1, annot=False, cmap='coolwarm', square=True, ax=axes[0, 0])
axes[0, 0].set_title(f'Correlation Matrix of MFCCs for speaker: {n1_speakers[speakers_idx[0]]} and digit: {n1}')

# Plot correlation matrix for wav file user1_n2
sns.heatmap(corr_matrix_user1_n2, annot=False, cmap='coolwarm', square=True, ax=axes[0, 1])
axes[0, 1].set_title(f'Correlation Matrix of MFCCs for speaker: {n2_speakers[speakers_idx[1]]} and digit: {n2}')

# Plot correlation matrix for wav file user2_n1
sns.heatmap(corr_matrix_user2_n1, annot=False, cmap='coolwarm', square=True, ax=axes[1, 0])
axes[1, 0].set_title(f'Correlation Matrix of MFCCs for speaker: {n1_speakers[speakers_idx[2]]} and digit: {n1}')

# Plot correlation matrix for wav file user2_n2
sns.heatmap(corr_matrix_user2_n2, annot=False, cmap='coolwarm', square=True, ax=axes[1, 1])
axes[1, 1].set_title(f'Correlation Matrix of MFCCs for speaker: {n2_speakers[speakers_idx[3]]} and digit: {n2}')

plt.tight_layout()
plt.show()

############################################################################################################
## BHMA 5
# Create the new list T by concatenating along the second axis
T = [np.concatenate((a, b, c), axis=0) for a, b, c in zip(mfccs, mfcc_deltas, mfcc_delta_deltas)]

# create mean, standard_deviation vector for every feature (13 MFCCs, 13 deltas, 13 delta_deltas)
def create_T_mean_sigma(T):
    T_mean_sigma = []
    dim = T[0].shape[0] # 39
    for array in T:
        # Initialize an array to store the mean and standard deviation tuples for the current array
        current_array = np.empty((dim,), dtype=(float, 2))
        for i in range(dim):
            current_array[i] = ( np.mean(array[i, :]), np.std(array[i, :]) )
        T_mean_sigma.append(current_array)
    return T_mean_sigma, dim

T_mean_sigma, num_features = create_T_mean_sigma(T) # 133 samples x 39 features x 2 (mean,sigma)

def mean_sigma_tolists(L,n=2):
    a,b,c,d = [],[],[],[]
    for array in L:
        a.append(array[0, 0])
        b.append(array[0, 1])
        c.append(array[1, 0])
        d.append(array[1, 1])
    return a,b,c,d

# we store the (mean,sigma) of the first two features
mean1,sigma1,mean2,sigma2 = mean_sigma_tolists(T_mean_sigma,2)

# Plot the means of the two first features
plt.figure(figsize=(13, 6))
ax = sns.scatterplot(x=mean1, y=mean2, hue=digits, style=digits, palette='tab10',legend='full')
ax.set_xlabel('MFCC1 Mean', fontsize=12)
ax.set_ylabel('MFCC2 Mean', fontsize=12)
ax.set_title('Mean of first two features', fontsize=16)
plt.legend(title='Digits', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Plot the standard deviations of the two first features
plt.figure(figsize=(13, 6))
ax = sns.scatterplot(x=sigma1, y=sigma2, hue=digits, style=digits, palette='tab10',legend='full')
ax.set_xlabel('MFCC1 Standard Deviation', fontsize=12)
ax.set_ylabel('MFCC2 Standard Deviation', fontsize=12)
ax.set_title('Standard Deviation of first two features', fontsize=16)
plt.legend(title='Digits', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()

############################################################################################################
# BHMA 6
# Using PCA we reduce the feature dimensions from 39 to 3
def apply_pca_to_data(data, dims=2):
    pca = PCA(n_components=dims)
    reduced_data = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return reduced_data, explained_variance_ratio

def split_mean_sigma(T,num_samples,num_features):
    means  = np.zeros( (num_samples,num_features) )
    sigmas = np.zeros( (num_samples,num_features) )
    for i in range(num_samples):
        tmp = T[i].T
        means[i]  = tmp[0,:]
        sigmas[i] = tmp[1,:]
    return means, sigmas

means, sigmas = split_mean_sigma(T_mean_sigma,num_samples,num_features)

means_reduced_2, explained_variance_ratio_mean  = apply_pca_to_data(means)
sigmas_reduced_2,explained_variance_ratio_sigma = apply_pca_to_data(sigmas)

# Plot the means of the 2 new PCA features
plt.figure(figsize=(13, 6))
ax = sns.scatterplot(x=means_reduced_2[:,0], y=means_reduced_2[:,1], hue=digits, style=digits, palette='tab10',
                     legend='full')
ax.set_xlabel('PCA1', fontsize=12)
ax.set_ylabel('PCA2', fontsize=12)
ax.set_title('PCA of Means, n=2', fontsize=16)
plt.legend(title='Digits', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Plot the standard deviations of the 2 new PCA features
plt.figure(figsize=(13, 6))
ax = sns.scatterplot(x=sigmas_reduced_2[:,0], y=sigmas_reduced_2[:,1], hue=digits, style=digits, palette='tab10',
                     legend='full')
ax.set_xlabel('PCA1', fontsize=12)
ax.set_ylabel('PCA2', fontsize=12)
ax.set_title('PCA of Standard Deviations, n=2', fontsize=16)
plt.legend(title='Digits', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()

print(f'Explained variance ratio of the mean from PCA1: {explained_variance_ratio_mean[0]}')
print(f'Explained variance ratio of the mean from PCA2: {explained_variance_ratio_mean[1]}')
print(f'Explained variance ratio of the standard deviation from PCA1: {explained_variance_ratio_sigma[0]}')
print(f'Explained variance ratio of the standard deviation from PCA2: {explained_variance_ratio_sigma[1]}')

# PCA, for n=3
means_reduced_3, explained_variance_ratio_mean_3  = apply_pca_to_data(means,3)
sigmas_reduced_3,explained_variance_ratio_sigma_3 = apply_pca_to_data(sigmas,3) # apply PCA to std

#dictionary to match alphanumeric to numeric for digits
digit_to_int     = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}
digit_int_labels = [digit_to_int[digit] for digit in digits]

# Plot the means
fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(means_reduced_3[:, 0], means_reduced_3[:, 1], means_reduced_3[:, 2], c=digit_int_labels,
                      cmap='tab10', depthshade=True)
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('3D Plot: PCA of Means')
legend1 = ax.legend(*scatter.legend_elements(), title="digits")
ax.add_artist(legend1)
plt.show()

# Plot the standard deviations
fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(sigmas_reduced_3[:, 0], sigmas_reduced_3[:, 1], sigmas_reduced_3[:, 2], c=digit_int_labels,
                      cmap='tab10', depthshade=True)
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('3D Plot: PCA of Standard Deviations')
legend1 = ax.legend(*scatter.legend_elements(), title="digits")
ax.add_artist(legend1)
plt.show()

print(f'Explained variance ratio of the mean from PCA1: {explained_variance_ratio_mean_3[0]}')
print(f'Explained variance ratio of the mean from PCA2: {explained_variance_ratio_mean_3[1]}')
print(f'Explained variance ratio of the mean from PCA3: {explained_variance_ratio_mean_3[2]}')
print(f'Explained variance ratio of the standard deviation from PCA1: {explained_variance_ratio_sigma_3[0]}')
print(f'Explained variance ratio of the standard deviation from PCA2: {explained_variance_ratio_sigma_3[1]}')
print(f'Explained variance ratio of the standard deviation from PCA3: {explained_variance_ratio_sigma_3[2]}')

############################################################################################################
# BHMA 7
# Standarize funnction
def scale_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def classify(means, sigmas):
    X = np.hstack([means,sigmas]) # 133 samples x 78 features (39 means + 39 sigmas)

    X_train, X_test, Y_train, Y_test = train_test_split(X, np.array(digits), test_size=0.3, random_state=42)

    # Scale data after splitting
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Naive Bayes Classifier
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Naive Bayes accuracy: {accuracy:.4f}')

    # kNN Classifier
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'kNN accuracy: {accuracy:.4f}')

    # Random Forest Classifier
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Random Forest accuracy: {accuracy:.4f}')

    # SVM Classifier
    svm = SVC(kernel='rbf', gamma='auto', random_state=42)
    svm.fit(X_train, Y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'SVM accuracy: {accuracy:.4f}')
    
classify(means, sigmas)

############################################################################################################
# Bonus
T_bonus = T.copy()
for i in range(num_samples):  
    
    zcr                = librosa.feature.zero_crossing_rate(wav_data[i], frame_length=window_in_samples, 
                                                                  hop_length=step_in_samples) 
    spectral_centroid  = librosa.feature.spectral_centroid(y=wav_data[i], sr=sampling_rate, n_fft=window_in_samples, 
                                      hop_length=step_in_samples)
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=wav_data[i], sr=sampling_rate, n_fft=window_in_samples, 
                                      hop_length=step_in_samples)
    spectral_contrast  = librosa.feature.spectral_contrast(y=wav_data[i], sr=sampling_rate, n_fft=window_in_samples, 
                                      hop_length=step_in_samples)
    chroma_features    = librosa.feature.chroma_stft(y=wav_data[i], sr=sampling_rate, n_fft=window_in_samples, 
                                      hop_length=step_in_samples)
    spectral_flatness  = librosa.feature.spectral_flatness(y=wav_data[i], n_fft=window_in_samples, 
                                      hop_length=step_in_samples)
    rmse               = librosa.feature.rms(y=wav_data[i], frame_length=window_in_samples, hop_length=step_in_samples)


    T_bonus[i] = np.vstack( (T_bonus[i], zcr, spectral_centroid, spectral_rolloff, spectral_contrast, chroma_features, spectral_flatness, rmse) )

T_bonus_mean_sigma, num_features = create_T_mean_sigma(T_bonus)
means, sigmas = split_mean_sigma(T_bonus_mean_sigma, num_samples, num_features)
classify(means, sigmas)

############################################################################################################
# BHMA 8
f = 40
num_sequences = 1000
num_points = 10
time_step = 1 / (f * num_points)
sin_seq, cos_seq = [],[]

for _ in range(num_sequences):
    
    # Random start time in [0,T]
    t_start   = np.random.uniform(0, 1 / f)
    time_step = np.random.uniform(0.001, 0.01)
    t         = t_start + np.arange(0, num_points) * time_step

    sine_wave = np.sin(2 * np.pi * f * t)
    sin_seq.append(sine_wave)

    cosine_wave = np.cos(2 * np.pi * f * t)
    cos_seq.append(cosine_wave)

sin_seq = np.array(sin_seq)
cos_seq = np.array(cos_seq)

# RNN model
class SineCosineRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SineCosineRNN, self).__init__()
        # Instantiate the layer 
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Linear transformation to the data: y = xA^T + b
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)

        # Forward propagation
        out, _ = self.rnn(x, h0)

        # The output is passed to the linear (fc) layer 
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 1 # Sequence of scalars, therefore 1
hidden_size = 50
output_size = 1
num_layers = 1

model = SineCosineRNN(input_size, hidden_size, output_size, num_layers)

sin_seq_tensor = torch.tensor(sin_seq, dtype=torch.float32).unsqueeze(-1)  # shape: (num_sequences, num_points, 1)
cos_seq_tensor = torch.tensor(cos_seq, dtype=torch.float32).unsqueeze(-1)  # shape: (num_sequences, num_points, 1)

train_size = int(0.7 * num_sequences)
val_size = int(0.15 * num_sequences)
test_size = num_sequences - train_size - val_size

sin_seq_train, sin_seq_val_test = torch.split(sin_seq_tensor, [train_size, val_size + test_size], dim=0)
cos_seq_train, cos_seq_val_test = torch.split(cos_seq_tensor, [train_size, val_size + test_size], dim=0)

sin_seq_val, sin_seq_test = torch.split(sin_seq_val_test, [val_size, test_size], dim=0)
cos_seq_val, cos_seq_test = torch.split(cos_seq_val_test, [val_size, test_size], dim=0)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(sin_seq_train)
    loss    = criterion(outputs, cos_seq_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():  # we only want to update gradients during training
        val_outputs = model(sin_seq_val)
        val_loss = criterion(val_outputs, cos_seq_val)
        val_losses.append(val_loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.3f}, Validation Loss: {val_loss.item():.3f}')

# Evaluate on test data
model.eval()
with torch.no_grad():
    predicted = model(sin_seq_test)
    test_loss = criterion(predicted, cos_seq_test)
    print(f'Test Loss: {test_loss.item():.3f}')
    # y_hat  = model(sin_seq_test).squeeze(-1).numpy()
    # y_true = cos_seq_test.squeeze(-1).numpy()


plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()