import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv(r"D:\Movies\Downloads\cleaned_student_dataset.csv")

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Course'] = le.fit_transform(data['Course'])
data['City'] = le.fit_transform(data['City'])

data['Result'] = np.where(data['Marks'] >= 50, 1, 0)

X = data[['Age', 'Attendance (%)', 'Gender', 'Course', 'City']]
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()