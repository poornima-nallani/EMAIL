
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv("emails.csv")

# Split features (X) and labels (y)
X = data.iloc[:, 1:-1].values  # Exclude the first column (Email name) and the last column (labels)
y = data.iloc[:, -1].values    # Labels are in the last column

# Split the dataset into training and testing sets, leaving 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions_svm = svm_classifier.predict(X_test)

# Calculate accuracy on the test data
accuracy_svm = accuracy_score(y_test, predictions_svm)
print("Accuracy:", accuracy_svm)

word_columns = data.columns[1:-1].values

# Initialize a list to store predictions for each email
predictions = []

# Initialize a list to store word count arrays
word_count_arrays = []


def tokenize_email(text):
    text = text.lower()
    # Tokenize the text using regular expressions
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def email_to_word_count_array(text):
    # Tokenize the email text
    words = tokenize_email(text)
    
    # Initialize an array to store word counts
    word_count_array = [0] * len(word_columns)
    # print(len(word_columns))
    # print('*******')
    # print(word_columns)
    # print('*******')
    # Count the occurrences of each word
    for word in words:
        if word in word_columns:
            index = np.where(word_columns == word)[0]
            if index.size > 0:
                word_count_array[index[0]] += 1

    # Trim the array to match the length of word_columns
    word_count_array = word_count_array[:len(word_columns)]
    
    return word_count_array

def predict_emails_svm(folder_path):
    # Initialize an empty list to store word count arrays for each email
    word_count_arrays = []

    # Get the list of files in the specified folder
    files = [f for f in os.listdir(folder_path) if f.startswith("email")]

    # Iterate over each file in the folder
    for file_name in files:
        # Read the contents of the email
        with open(os.path.join(folder_path, file_name), 'r') as file:
            email_content = file.read()
            # Convert the email text into a word count array
            word_count_array = email_to_word_count_array(email_content)
            word_count_arrays.append(word_count_array)          

    # Convert the list of word count arrays to a numpy array
    X_test_new = np.array(word_count_arrays)
    # print(X_test_new.shape)

    # Make predictions on the test data
    predictions = svm_classifier.predict(X_test_new)
    return predictions
    
ans = predict_emails_svm("test")
print(ans)




