import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from sklearn.feature_selection import SelectKBest, chi2
from keras.regularizers import l1
from keras.metrics import BinaryAccuracy
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df0 = pd.read_csv('Cleaveland_heart_disease_dataset.csv')

# Display the first few rows of the dataset
print(df0.head())
print(df0.info())
shape = df0.shape

# Removing unused and undefined values
df = df0.drop(columns=["Id", "ccf", "name", "junk", "cathef", "lmt", "ladprox", "laddist", "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "trestbps.1", "pncaden", "htn", "restckm", "exerckm", "xhypo"])

# Missing values check
missing_values = df.isnull().sum()

# Distribution check
target_distribution = df['num'].value_counts(normalize=True) * 100

print(shape, missing_values[missing_values > 0], target_distribution)

# Convert target variable to binary
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

print("target variable count")
print(df['num'].value_counts())

# Replace placeholder value (-9) with NaN
df.replace(-9, float('nan'), inplace=True)


# Handle missing values
for col in df.columns:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

# Normalization
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Splitting the dataset into train and test sets
X = df_scaled.drop('num', axis=1)
y = df['num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, X_test.shape)

"""# Applying SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Distribution after SMOTE:")
print(y_train.value_counts())"""

# Feature selection
selector = SelectKBest(chi2, k=14)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print("Selected features using chi-squared method:")
print(selected_features)

"""# plotting  the heatmap
subset = df_scaled[selected_features]
corr = subset.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black')
plt.title('Correlation heatmap of selected features')
plt.savefig('heatmap.png', dpi=300)"""

# Descriptive statistics for selected features
desc_stats = df[selected_features].describe()
print("Descriptive stats")
print(desc_stats)

# Inferential statistics - Correlation matrix
infer_stats = df[selected_features].corr()
print("Inferential statistics")
print(infer_stats)

# Creating LSTM model with L1 regularization
def create_L1_reg_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu', activity_regularizer=l1(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    return model

# Train the model
input_shape = (X_train_selected.shape[1], 1)
model = create_L1_reg_lstm_model(input_shape)
X_train_chisq_reshaped = np.reshape(X_train_selected, (X_train_selected.shape[0], X_train_selected.shape[1], 1))
model.fit(X_train_chisq_reshaped, y_train, epochs=50, batch_size=64)
X_test_chisq_reshaped = np.reshape(X_test_selected, (X_test_selected.shape[0], X_test_selected.shape[1], 1))
loss, accuracy, auc = model.evaluate(X_test_chisq_reshaped, y_test, verbose=0)

# Evaluation metrics
print(f"LSTM Accuracy: {accuracy * 100:.2f}%")
print(f"LSTM auc: {auc * 100:.2f}")
print(f"LSTM Loss: {loss * 100:.2f}%")

# Generate confusion matrix
y_pred_probs_lstm = model.predict(X_test_chisq_reshaped)
y_pred_labels_lstm = np.round(y_pred_probs_lstm).astype(int)

conf_matrix_lstm = confusion_matrix(y_test, y_pred_labels_lstm)

print(f"LSTM Confusion matrix")
print(conf_matrix_lstm)

# Creating the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

# Reshaping input to be 3D (samples, time steps, features)
X_train_cnn = X_train_selected.reshape(X_train_selected.shape[0], X_train_selected.shape[1], 1)
X_test_cnn = X_test_selected.reshape(X_test_selected.shape[0], X_test_selected.shape[1], 1)

# CNN model
cnn_model = create_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2]))
history_cnn = cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=64, validation_data=(X_test_cnn, y_test), verbose=0)

# Evaluation metrics
cnn_loss, cnn_accuracy, cnn_auc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print("CNN Accuracy: {:.2f}%".format(cnn_accuracy * 100))
print("CNN AUC: {:.2f}%".format(cnn_auc * 100))
print("CNN loss: {:.2f}%".format(cnn_loss * 100))

# Generate confusion matrix
y_pred_probs_cnn = model.predict(X_test_cnn)
y_pred_labels_cnn = np.round(y_pred_probs_cnn).astype(int)

conf_matrix_cnn = confusion_matrix(y_test, y_pred_labels_cnn)

print(f"CNN Confusion matrix")
print(conf_matrix_cnn)

# Creating the FF-NN
def create_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# FF-NN model
nn_model = create_nn_model(X_train_selected.shape[1])
history_nn = nn_model.fit(X_train_selected, y_train, epochs=50, batch_size=64, validation_data=(X_test_selected, y_test), verbose=0)

# Evaluation metrics
nn_loss, nn_accuracy, nn_auc = nn_model.evaluate(X_test_selected, y_test, verbose=0)
print("FF Neural Network Accuracy: {:.2f}%".format(nn_accuracy * 100))
print("FF Neural Network auc: {:.2f}%".format(nn_auc * 100))
print("FF Neural Network loss: {:.2f}%".format(nn_loss * 100))

# Creating a simple RNN model
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=input_shape, return_sequences=True))
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

# Reshape input
X_train_rnn = X_train_selected.reshape(X_train_selected.shape[0], X_train_selected.shape[1], 1)
X_test_rnn = X_test_selected.reshape(X_test_selected.shape[0], X_test_selected.shape[1], 1)

# RNN model
rnn_model = create_rnn_model((X_train_rnn.shape[1], X_train_rnn.shape[2]))
history_rnn = rnn_model.fit(X_train_rnn, y_train, epochs=50, batch_size=64, validation_data=(X_test_rnn, y_test), verbose=0)

# Evaluation metrics
rnn_loss, rnn_accuracy, rnn_auc = rnn_model.evaluate(X_test_rnn, y_test, verbose=0)
print("RNN Accuracy: {:.2f}%".format(rnn_accuracy * 100))
print("RNN auc: {:.2f}%".format(rnn_auc * 100))
print("RNN loss: {:.2f}%".format(rnn_loss * 100))

# Generate predictions
y_pred_probs_rnn = model.predict(X_test_rnn)
y_pred_labels_rnn = np.round(y_pred_probs_rnn).astype(int)

# Generate the confusion matrix
conf_matrix_rnn = confusion_matrix(y_test, y_pred_labels_rnn)

print(f"RNN Confusion matrix")
print(conf_matrix_rnn)
