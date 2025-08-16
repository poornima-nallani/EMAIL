
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class GaussianNaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)

        # Initialize dictionaries to store mean and variance for each class and feature
        self.mean = {}
        self.var = {}
        self.class_prior = {}

        # Compute class priors
        for c in self.classes:
            self.class_prior[c] = np.mean(y_train == c)

        # Compute mean and variance for each class and feature
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)

    def _pdf(self, x, mean, var):
        epsilon = 1e-6  # Small epsilon value to prevent division by zero
        var += epsilon  # Add epsilon to the variance
        exponent = -((x - mean) ** 2) / (2 * var)
        pdf = (1 / (np.sqrt(2 * np.pi * var))) * np.exp(exponent)
        return np.where(var == 0, 1, pdf)


    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            posteriors = []

            # Compute posterior for each class
            for c in self.classes:
                prior = np.log(self.class_prior[c])
                likelihood = np.sum(np.log(self._pdf(x, self.mean[c], self.var[c])))
                posterior = prior + likelihood
                posteriors.append(posterior)

            # Assign the class with maximum posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])

        return y_pred




# Load the dataset
data = pd.read_csv("emails.csv")

# Split features (X) and labels (y)
X = data.iloc[:, 1:-1].values  # Exclude the first column (Email name) and the last column (labels)
y = data.iloc[:, -1].values    # Labels are in the last column

word_columns = data.columns[1:-1]


# Split the dataset into training and testing sets, leaving 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gaussian Naive Bayes classifier
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Make predictions on the test data
predictions = gnb.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

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
  
    for word in words:
        if word in word_columns:
            index = np.where(word_columns == word)[0]
            if index.size > 0:
                word_count_array[index[0]] += 1

    # Trim the array to match the length of word_columns
    word_count_array = word_count_array[:len(word_columns)]
    
    return word_count_array

def predict_emails_gnb(folder_path):
    # Get the list of files in the specified folder
    files = [f for f in os.listdir(folder_path) if f.startswith("email")]

    # Initialize a list to store predictions for each email
    predictions = []

    # Iterate over each file in the folder
    for file_name in files:
        # Read the contents of the email
        with open(os.path.join(folder_path, file_name), 'r') as file:
            email_content = file.read()
            # print(email_content)
            # Convert the email text into a word count array
            word_count_array = email_to_word_count_array(email_content)
            # print(word_count_array)
            word_count_arrays.append(word_count_array)          

    X_test_new = np.array(word_count_arrays)

    # Make predictions on the test data
    predictions = gnb.predict(X_test_new)
    return predictions
    
ans = predict_emails_gnb("test")
print(ans)

