#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Introduction to Machine Learning - 25737-2</h1>
# <h4 align="center">Dr. R. Amiri</h4>
# <h4 align="center">Sharif University of Technology, Spring 2024</h4>
# 
# 
# **<font color='red'>Plagiarism is strongly prohibited!</font>**
# 
# 
# **Student Name**: Ehsan Merrikhi
# 
# **Student ID**: 400101967
# 
# 
# 
# 

# # Logistic Regression

# **Task:** Implement your own Logistic Regression model, and test it on the given dataset of Logistic_question.csv!

# In[1]:


import numpy as np

class MyLogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(z):
        nom = 1
        denom = 1 + np.exp(-z)
        return nom / denom

    @staticmethod
    def loss_function(y, y_pred):
        log_loss = np.mean(-y * np.log(y_pred) -(1-y) * np.log(1-y_pred))
        return log_loss

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = MyLogisticRegression.sigmoid(linear_model)

            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = MyLogisticRegression.sigmoid(linear_model)
        return np.round(y_pred)


# **Task:** Test your model on the given dataset. You must split your data into train and test, with a 0.2 split, then normalize your data using X_train data. Finally, report 4 different evaluation metrics of the model on the test set. (You might want to first make the Target column binary!)

# In[4]:


# Your code goes here!
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# get data
data = pd.read_csv('Logistic_question.csv')
X = data.drop('Target', axis=1)
y = data['Target']
y = (y > 0.5)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# normalize
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# do office work
model = MyLogisticRegression()
model.fit(X_train_norm, y_train)
y_pred = model.predict(X_test_norm)

# how good was my accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# doesn't seem bad
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1-score: {f1:.6f}")


# **Question:** What are each of your used evaluation metrics? And for each one, mention situations in which they convey more data on the model performance in specific tasks.

# *Accuracy:* accuracy shows the proportion of right predictions out of total predictions. It shows how good the model works in general.
# 
# *Precision:* precision shows the proportion of true positive predictions out of the total positive predictions. Precision is important when the cost of a false positive is high.
# 
# *Recall:* recall shows the proportion of true positive predictions out of the total positive instances. Recall is important when the cost of a false negative is high.
# 
# *F1-score:* The mean of precision and recall, The F1-score is useful when you want to have a single metric that captures both precision and recall.

# **Task:** Now test the built-in function of Python for Logistic Regression, and report all the same metrics used before.

# In[3]:


from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Logistic_question.csv')
X = data.drop('Target', axis=1)
y = data['Target']
y = (y > 0.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# normalize
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# do office work
model = LogisticRegression()
model.fit(X_train_norm, y_train)
y_pred = model.predict(X_test_norm)

# how good was my accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# doesn't seem bad
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1-score: {f1:.6f}")


# **Question:** Compare your function with the built-in function. On the matters of performance and parameters. Briefly explain what the parameters of the built-in function are and how they affect the model's performance?

# *My code metrics report:* 
# 
# Accuracy: 0.850000 
# 
# Precision: 0.850000 
# 
# Recall: 1.000000 
# 
# F1-score: 0.918919 
# 
# *Built in library metrics report:* 
# Accuracy: 0.900000 
# 
# Precision: 0.894737
# 
# Recall: 1.000000
# 
# F1-score: 0.944444
# 
# 
# we see in every aspect the built in model was better than mine.
# 
# also the execution time for my model was higher than the sklearn library.
#     

# # Multinomial Logistic Regression

# **Task:** Implement your own Multinomial Logistic Regression model. Your model must be able to handle any number of labels!

# In[8]:


import numpy as np

class MyMultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100, lambda_=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.theta = None
        self.num_classes = None

    def softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, X, y):
        m, n = X.shape
        self.num_classes = len(np.unique(y))
        self.theta = np.zeros((n + 1, self.num_classes))

        # Add a column of ones to X for bias term
        X_bias = np.hstack((X, np.ones((m, 1))))

        for _ in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X_bias, self.theta)
            y_pred = self.softmax(linear_model)

            # Compute gradients
            one_hot_y = np.eye(self.num_classes)[y]
            error = y_pred - one_hot_y
            gradient = np.dot(X_bias.T, error) / m

            # Update parameters
            self.theta -= self.learning_rate * gradient

    def predict_probability(self, X):
        # Add a column of ones to X for bias term
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        scores = np.dot(X_bias, self.theta)
        return self.softmax(scores)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        return np.argmax(probabilities, axis=1)


# **Task:** Test your model on the given dataset. Do the same as the previous part, but here you might want to first make the Target column quantized into $i$ levels. Change $i$ from 2 to 10.

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Logistic_question.csv')
X = data.drop('Target', axis=1)
y = data['Target']

i = 4
bins = np.linspace(0, 1, i)
y = np.digitize(y, bins)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
# normalize
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
# do office work
model = MyMultinomialLogisticRegression(i)
model.fit(X_train_norm, y_train)
y_pred = model.predict(X_test_norm)
print(y_pred)
# how good was my accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
# doesn't seem bad
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1-score: {f1:.6f}")


# **Question:** Report for which $i$ your model performs best. Describe and analyze the results! You could use visualizations or any other method!

# **Your answer:**

# # Going a little further!

# First we download Adult income dataset from Kaggle! In order to do this create an account on this website, and create an API. A file named kaggle.json will be downloaded to your device. Then use the following code:

# In[ ]:


from google.colab import files 
files.upload()  # Use this to select the kaggle.json file from your computer
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# Then use this code to automatically download the dataset into Colab.

# In[ ]:


get_ipython().system('kaggle datasets download -d wenruliu/adult-income-dataset')
get_ipython().system('unzip /content/adult-income-dataset.zip')


# **Task:** Determine the number of null entries!

# In[13]:


# Your code goes here!

import pandas as pd
# null entries in the dataset are filled with "?"
# we need to find how many "?" we have
data = pd.read_csv('adult.csv')
data.replace('?', pd.NA, inplace=True)
null_count = data.isnull().sum().sum()

print(null_count)


# **Question:** In many widely used datasets there are a lot of null entries. Propose 5 methods by which, one could deal with this problem. Briefly explain how do you decide which one to use in this problem.

# **Your answer:**
# 
# *1:*
# we can delete all the rows that have at least one null entry.
# 
# *2:*
# we can replace null entries with median or mode or mean of other valid values.
# 
# *3:*
# we can also use machine learning methods to predict the value which is missing.
# 
# *4:*
# if null entries could contain information we can also use them as valid values.
# 
# *5:*
# there are algorithms that support missing values so we can use them instead.

# **Task:** Handle null entries using your best method.

# In[14]:


# Your code goes here!
# decided to replace NA values with mode of column
for column in data.columns:
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)


# **Task:** Convert categorical features to numerical values. Split the dataset with 80-20 portion. Normalize all the data using X_train. Use the built-in Logistic Regression function and GridSearchCV to train your model, and report the parameters, train and test accuracy of the best model.

# In[19]:


# Your code goes here!
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str))
X = data.drop('income', axis=1)
Y = data['income']
# print(data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=67)

# print(X_train.shape)
# print(X_test.shape)
# print(data.shape)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

log_reg_model = LogisticRegression()
log_reg_model.fit(X=X_train_norm, y=y_train)
y_pred_log = log_reg_model.predict(X_test_norm)
accuracy_log = accuracy_score(y_test, y_pred_log)
param_log = log_reg_model.coef_


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],  
    'max_iter': [100, 200, 300]  
}

grid_search_model = GridSearchCV(LogisticRegression(), param_grid=param_grid)
grid_search_model.fit(X_train_norm, y_train)

best_param = grid_search_model.best_params_
best_model = grid_search_model.best_estimator_

train_accuracy = accuracy_score(y_train, best_model.predict(X_train_norm))
test_accuracy = accuracy_score(y_test, best_model.predict(X_test_norm))

print(f'best_param: {best_param}')
print(f'Train Accuracy: {100 * train_accuracy:.4f}%')
print(f'Test Accuracy: {100 * test_accuracy:.4f}%')


# **Task:** To try a different route, split X_train into $i$ parts, and train $i$ separate models on these parts. Now propose and implement 3 different *ensemble methods* to derive the global models' prediction for X_test using the results(not necessarily predictions!) of the $i$ models. Firstly, set $i=10$ to find the method with the best test accuracy(the answer is not general!). You must Use your own Logistic Regression model.(You might want to modify it a little bit for this part!)

# In[ ]:





# **Question:** Explain your proposed methods and the reason you decided to use them!

# **Your answer:**

# **Task:** Now, for your best method, change $i$ from 2 to 100 and report $i$, train and test accuracy of the best model. Also, plot test and train accuracy for $2\leq i\leq100$.

# In[ ]:


# Your code goes here!


# **Question:** Analyze the results.

# **Your Answer:**
