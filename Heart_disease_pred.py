import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Input
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
chd_data = pd.read_csv("Cleaveland_heart_disease_dataset.csv")
print(chd_data.head())
print(chd_data.describe())

# Removing unused and undefined values
data = chd_data.drop(columns=["Id", "ccf", "name", "junk", "cathef", "lmt", "ladprox", "laddist", "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "trestbps.1", "pncaden", "htn"])

data.replace(-9, np.nan, inplace=True)
missing_values = data.isnull().sum()
print("Missing Values per Column:")
print(missing_values)

# replace missing values
mode_painloc = data["painloc"].mode()[0]
data["painloc"].fillna(mode_painloc, inplace=True)

mode_painexer = data["painexer"].mode()[0]
data["painexer"].fillna(mode_painexer, inplace=True)

mode_relrest = data["relrest"].mode()[0]
data["relrest"].fillna(mode_relrest, inplace=True)

mode_cp = data["cp"].mode()[0]
data["cp"].fillna(mode_cp, inplace=True)

mean_trestbps = round(data["trestbps"].mean())
data["trestbps"].fillna(mean_trestbps, inplace=True)

mean_chol = data["chol"].mean()
data["chol"].fillna(f"{mean_chol:.3f}", inplace=True)

mean_cigs = round(data["cigs"].mean())
data["cigs"].fillna(mean_cigs, inplace=True)

data.loc[data["cigs"] > 0, "smoke"] = 1

data.loc[data["smoke"] == 1, "years"] = round(data["years"].mean())

mode_fbs = data["fbs"].mode()[0]
data["fbs"].fillna(mode_fbs, inplace=True)

mode_famhist = data["famhist"].mode()[0]
data["famhist"].fillna(mode_famhist, inplace=True)

mode_restecg = data["restecg"].mode()[0]
data["restecg"].fillna(mode_restecg, inplace=True)

mean_ekgmo = round(data["ekgmo"].mean())
data["ekgmo"].fillna(mean_ekgmo, inplace=True)

mean_ekgmo = round(data["ekgmo"].mean())
data["ekgmo"].fillna(mean_ekgmo, inplace=True)

mean_ekgday = round(data["ekgday"].mean())
data["ekgday"].fillna(mean_ekgday, inplace=True)

mode_ekgyr = data["ekgyr"].mode()[0]
data["ekgyr"].fillna(mode_ekgyr, inplace=True)

mode_dig = data["dig"].mode()[0]
data["dig"].fillna(mode_dig, inplace=True)

mode_prop = data["prop"].mode()[0]
data["prop"].fillna(mode_prop, inplace=True)

mode_nitr = data["nitr"].mode()[0]
data["nitr"].fillna(mode_nitr, inplace=True)

mode_pro = data["pro"].mode()[0]
data["pro"].fillna(mode_pro, inplace=True)

mode_diuretic = data["diuretic"].mode()[0]
data["diuretic"].fillna(mode_diuretic, inplace=True)

mode_proto = data["proto"].mode()[0]
data["proto"].fillna(mode_proto, inplace=True)

mean_thaldur = data["thaldur"].mean()
data["thaldur"].fillna(f"{mean_thaldur:.1f}", inplace=True)

mean_thaltime = data["thaltime"].mean()
data["thaltime"].fillna(f"{mean_thaltime:.1f}", inplace=True)

mean_met = round(data["met"].mean())
data["met"].fillna(mean_met, inplace=True)

mean_thalach = round(data["thalach"].mean())
data["thalach"].fillna(mean_thalach, inplace=True)

mean_thalrest = round(data["thalrest"].mean())
data["thalrest"].fillna(mean_thalrest, inplace=True)

mean_tpeakbps = round(data["tpeakbps"].mean())
data["tpeakbps"].fillna(mean_tpeakbps, inplace=True)

mean_tpeakbpd = round(data["tpeakbpd"].mean())
data["tpeakbpd"].fillna(mean_tpeakbpd, inplace=True)

mean_trestbpd = round(data["trestbpd"].mean())
data["trestbpd"].fillna(mean_trestbpd, inplace=True)

mode_exang = data["exang"].mode()[0]
data["exang"].fillna(mode_exang, inplace=True)

mode_xhypo = data["xhypo"].mode()[0]
data["xhypo"].fillna(mode_xhypo, inplace=True)

mean_oldpeak = data["oldpeak"].mean()
data["oldpeak"].fillna(f"{mean_oldpeak:.1f}", inplace=True)

mode_slope = data["slope"].mode()[0]
data["slope"].fillna(mode_slope, inplace=True)

mean_rldv5 = round(data["rldv5"].mean())
data["rldv5"].fillna(mean_rldv5, inplace=True)

mean_rldv5e = round(data["rldv5e"].mean())
data["rldv5e"].fillna(mean_rldv5e, inplace=True)

mode_ca = data["ca"].mode()[0]
data["ca"].fillna(mode_ca, inplace=True)



# Setting as the target variable
X = data.drop(columns=["num"])
# X = data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
y = data["num"]

# SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_balanced)

"""# Apply PCA
num_components = 13  # You can adjust the number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_scaled)
selected_feature_indices = np.argsort(pca.components_)[::-1][:num_components]
selected_features = [X.columns[idx] for idx in selected_feature_indices]
print("Selected Features after PCA:", selected_features)"""

# Apply Chi-Squared feature selection
num_features_to_select = 13  # You can adjust the number of features
chi2_selector = SelectKBest(chi2, k=num_features_to_select)
X_chi2_selected = chi2_selector.fit_transform(X_scaled, y_balanced)
selected_feature_indices = chi2_selector.get_support(indices=True)
selected_features = [X.columns[idx] for idx in selected_feature_indices]
print("Selected chi-sq Features:", selected_features)

# Logistic Regression model for RFE implementation
lr_model = LogisticRegression()
# lr_model = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='auto', n_jobs=None, penalty='l2', random_state=1234, solver='lbfgs', tol=0.0001, verbose=0,warm_start=False)

# Decision tree model
dt_model = tree.DecisionTreeClassifier()

# Random forest model
rf_model = RandomForestClassifier(max_depth=24, random_state=0)

# SVM model
svm_model = SVC()

# Define CNN model
cnn_model = Sequential()
#cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_pca.shape[1], 1)))
#cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_scaled.shape[1], 1)))
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_chi2_selected.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_chi2_selected.shape[1], 1)))
#lstm_model.add(LSTM(64, input_shape=(X_pca.shape[1], 1)))
#lstm_model.add(LSTM(64, input_shape=(X_scaled.shape[1], 1)))

# Concatenate both models
merged = concatenate([cnn_model.output, lstm_model.output])

# Additional dense layers for ensemble
ensemble_layers = Dense(32, activation='relu')(merged)
ensemble_layers = Dense(1, activation='sigmoid')(ensemble_layers)

# Create ensemble model
ensemble_model = Model(inputs=[cnn_model.input, lstm_model.input], outputs=ensemble_layers)
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


"""# LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_pca.shape[1], 1)))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"""

# K-Fold Cross-Validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

#for fold, (train_index, test_index) in enumerate(kf.split(X_pca)):
for fold, (train_index, test_index) in enumerate(kf.split(X_chi2_selected)):
#for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    print(f"Fold: {fold+1}")

    #X_train, X_test = X_pca[train_index], X_pca[test_index]
    #X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    X_train, X_test = X_chi2_selected[train_index], X_chi2_selected[test_index]
    y_train, y_test = y_balanced[train_index], y_balanced[test_index]

    # Logistic Regression
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    # Decision Tree
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred_dt))

    # Random Forest
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # SVM Model
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))

    # Reshape data for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Reshape data for CNN
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Ensemble model
    ensemble_model.fit([X_train_cnn, X_train_lstm], y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = ensemble_model.evaluate([X_test_cnn, X_test_lstm], y_test)
    print(f"Ensemble Model Accuracy: {accuracy}")

    """# LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
    _, accuracy = lstm_model.evaluate(X_test_lstm, y_test)
    print(f"LSTM Accuracy: {accuracy}")"""

    print("-" * 40)
