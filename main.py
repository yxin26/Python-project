pip install streamlit


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics

# Load datasets and models
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))
X_train = pickle.load(open('X_train.sav', 'rb'))

# Dictionary for labels
dic = {0: 'Bad', 1: 'Good'}

# Function to test a certain index of the dataset
def test_demo(index):
    values = X_test.iloc[index]  # Access the row by index
    input_data = []

    # Creating appropriate input widget based on data type
    for col in X_test.columns:
        if pd.api.types.is_numeric_dtype(X_test[col]):
            # It's a numeric column, create a slider
            value = values[col]
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            step = (max_val - min_val) / 100
            # Convert numpy types to Python native types for Streamlit compatibility
            slider_val = st.sidebar.slider(col, float(min_val), float(max_val), float(value), float(step))
            input_data.append(slider_val)
        else:
            # It's a categorical column, create a selectbox
            options = list(X_test[col].unique())
            selected = st.sidebar.selectbox(col, options, index=options.index(values[col]))
            # One-hot encode the selected option
            encoded_features = pd.get_dummies(X_test[col], prefix=col)
            input_features = encoded_features.loc[:, encoded_features.columns == f"{col}_{selected}"].iloc[0].tolist()
            input_data.extend(input_features)

    input_df = pd.DataFrame([input_data], columns=X_test.columns)

    classifier = st.selectbox('Which algorithm?', ['Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree', 'SVM'])
    model_path = f'best_{classifier.lower().replace(" ", "_")}_model.sav'
    model = pickle.load(open(model_path, 'rb'))
    res = model.predict(input_df)[0]
    st.write('Prediction: ', dic[res])

    pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, pred)
    st.write('Accuracy: ', score)
    st.write('Confusion Matrix: ', cm)

# Application title and interaction
st.title('HELOC Prediction Interface')

if st.checkbox('Show dataframe'):
    st.write(X_test)

index = st.text_input('Enter a row index to predict from X_test:', '0')
if index.isdigit():
    index = int(index)
    if 0 <= index < len(X_test):
        test_demo(index)
    else:
        st.error('Index out of range.')
else:
    st.error('Please enter a valid index.')
