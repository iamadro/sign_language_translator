import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklmt
import numpy as np


data_dict = pickle.load(open('./data.pkl', 'rb'))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = sklmt.accuracy_score(y_predict, y_test)
prec = sklmt.precision_score(y_test, y_predict, average='macro')
rec = sklmt.recall_score(y_test, y_predict, average='macro')
score = sklmt.accuracy_score(y_test, y_predict)
f1score = sklmt.f1_score(y_test, y_predict, average='macro')

print("Precision: ", prec*100)
print("Recall: ", rec*100)
print("Accuracy Score: ", score*100)
print("F1 Score: ", f1score*100)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('useless_model.pkl', 'wb')
pickle.dump({'model': model}, f)
f.close()
