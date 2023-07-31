import pandas as pd
import numpy as np
from mlxtend.evaluate import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics._plot.confusion_matrix import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
import plotly.figure_factory as ff

chd_data = pd.read_csv("Cleaveland_heart_disease_dataset - Copy.csv")
print(chd_data.head())
print(chd_data.describe())

data = chd_data.drop(columns=["name"])

# Na value count in each column
print("Number of Na values in each column:")
print(data.isnull().sum())

# Drop rows with Na values
chd_data.dropna(inplace=True)

print(data.head())

corrs = data.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
#print(figure.show())

#X = data.drop(columns=["num"])
X = data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
y = data["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model for RFE implementation
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

y_Pred_lr = lr_model.predict(X_test_scaled)
print(y_Pred_lr)
cm = confusion_matrix(y_test, y_Pred_lr, labels=lr_model.classes_)
print("Confusion matrix:\n")
print(cm)
print("LogisticRegression classification_report:\n")
print(classification_report(y_test, y_Pred_lr))

# Decision tree model
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)

y_Pred_dt = dt_model.predict(X_test_scaled)
print(y_Pred_dt)
cm = confusion_matrix(y_test, y_Pred_dt, labels=dt_model.classes_)
print("Confusion matrix:\n")
print(cm)
print("Decision tree classification_report:\n")
print(classification_report(y_test, y_Pred_dt))

# Random forest model
rf_model = RandomForestClassifier(max_depth=24, random_state=0)
rf_model.fit(X_train_scaled, y_train)

y_Pred_rf = rf_model.predict(X_test_scaled)
print(y_Pred_rf)
cm = confusion_matrix(y_test, y_Pred_rf, labels=rf_model.classes_)
print("Confusion matrix:\n")
print(cm)
print("Random forest classification_report:\n")
print(classification_report(y_test, y_Pred_rf))

"""
# Indirect implementation of RFE with Logistic Regression model as the estimator and the number of features to select
num_features_to_select = 15
rfe_selector = RFE(estimator=lr_model, n_features_to_select=num_features_to_select)

# Fit RFE
rfe_selector.fit(X_train_scaled, y_train)

# Getting the selected features
selected_features_mask = rfe_selector.support_
selected_features = X.columns[selected_features_mask]

print("Selected features:", selected_features)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# Reshaping data
X_train_reshaped = X_train_selected.values.reshape((X_train_selected.shape[0], 1, X_train_selected.shape[1]))
X_test_reshaped = X_test_selected.values.reshape((X_test_selected.shape[0], 1, X_test_selected.shape[1]))
"""

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# Create an LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training + evaluation of the LSTM model
model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=1)
accuracy = model.evaluate(X_test_reshaped, y_test)[1]
print("LSTM Accuracy:", accuracy)
