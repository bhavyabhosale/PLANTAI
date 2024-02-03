import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import pickle
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import os  # Import the os module for working with file paths

warnings.filterwarnings('ignore')
sns.set_style("whitegrid", {'axes.grid': False})
pio.templates.default = "plotly_white"


# Function to print information about the dataset
def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\n')
    print('Dataset columns:', df.columns)
    print('\n')
    print('Data types of each column: ', df.info())


# Function to check and remove duplicates
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')


# Function to read in and split data
def read_in_and_split_data(data, target):
    # Handle NaN values
    data = data.dropna(subset=[target])

    if data.shape[0] < 2:
        raise ValueError("Not enough samples after data preprocessing. Adjust handling of NaN values or use a larger dataset.")

    X = data.drop(target, axis=1)
    y = data[target]

    # Impute NaN values with the mean (you can adjust this strategy)
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# Function to spot-check algorithms
def get_models():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(probability=True)))
    return models


# Function to ensemble models
def ensemble_models():
    ensembles = []
    ensembles.append(('AB', AdaBoostClassifier()))
    ensembles.append(('GBM', GradientBoostingClassifier()))
    ensembles.append(('RF', RandomForestClassifier()))
    ensembles.append(('Bagging', BaggingClassifier()))
    ensembles.append(('ET', ExtraTreesClassifier()))
    return ensembles


# Function to spot-check normalized models
def normalized_model(name_of_scaler):
    if name_of_scaler == 'standard':
        scaler = StandardScaler()
    elif name_of_scaler == 'minmax':
        scaler = MinMaxScaler()
    elif name_of_scaler == 'normalizer':
        scaler = Normalizer()
    elif name_of_scaler == 'binarizer':
        scaler = Binarizer()

    pipelines = []
    pipelines.append((name_of_scaler + 'LR', Pipeline([('Scaler', scaler), ('LR', LogisticRegression())])))
    pipelines.append((name_of_scaler + 'LDA', Pipeline([('Scaler', scaler), ('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append((name_of_scaler + 'KNN', Pipeline([('Scaler', scaler), ('KNN', KNeighborsClassifier())])))
    pipelines.append((name_of_scaler + 'CART', Pipeline([('Scaler', scaler), ('CART', DecisionTreeClassifier())])))
    pipelines.append((name_of_scaler + 'NB', Pipeline([('Scaler', scaler), ('NB', GaussianNB())])))
    pipelines.append((name_of_scaler + 'SVM', Pipeline([('Scaler', scaler), ('SVM', SVC())])))
    pipelines.append((name_of_scaler + 'AB', Pipeline([('Scaler', scaler), ('AB', AdaBoostClassifier())])))
    pipelines.append((name_of_scaler + 'GBM', Pipeline([('Scaler', scaler), ('GMB', GradientBoostingClassifier())])))
    pipelines.append((name_of_scaler + 'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier())])))
    pipelines.append((name_of_scaler + 'ET', Pipeline([('Scaler', scaler), ('ET', ExtraTreesClassifier())])))

    return pipelines


# Function to fit models and print cross-validation results
def fit_model(X_train, y_train, models):
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return names, results


# Function to save trained model
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


# Function to print classification metrics
def classification_metrics(model, conf_matrix, X_train, y_train, X_test, y_test, y_pred):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))


# Load Dataset
df = pd.read_csv('SmartCrop-Dataset.csv')

# Identify and handle non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
df[non_numeric_columns] = df[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remove Outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Check and remove duplicates
checking_removing_duplicates(df_out)

# Split Data to Training and Validation set
target = 'label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df_out, target)

# Train model
pipeline = make_pipeline(StandardScaler(), GaussianNB())
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_metrics(model, conf_matrix, X_train, y_train, X_test, y_test, y_pred)

# Save model
save_model(model, 'model.pkl')
