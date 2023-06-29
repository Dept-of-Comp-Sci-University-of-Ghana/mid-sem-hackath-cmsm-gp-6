import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# read dataset
df = pd.read_json("data.json")

# position + experience
df['pos_exp'] = df['position'] + ' ' + df['Experience'] + ' ' + df['skills']


training_data, testing_data = train_test_split(df, random_state=0, test_size=0.2)

# GET LABELS
y_train = training_data['category'].values
y_test = testing_data['category'].values

cv = CountVectorizer(binary=True, max_df=0.95)
cv.fit_transform(training_data['pos_exp'].values)

X_train = cv.transform(training_data['pos_exp'].values)
X_test = cv.transform(testing_data['pos_exp'].values)

log_reg = LogisticRegression()

# train data
model = log_reg.fit(X_train, y_train)

# make prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)

# print("Logistic Regression", score)