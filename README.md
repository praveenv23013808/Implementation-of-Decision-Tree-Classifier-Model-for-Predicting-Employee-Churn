# EX 06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## DATE: 13-03-2024
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sriram.V
RegisterNumber:  212222103002
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Data.Head():

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/0bd095a7-4fbb-478a-9f96-26c38fbbf0da)

Data.info():

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/4487cbaf-c2c6-4ec8-91c2-b019e0e58588)

isnull() and sum():

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/3ffc0e5d-bc8d-48be-950c-a6568170218f)

Data.Head() for salary:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/9213e19a-ca76-40ad-96db-a7368ff75905)

MSE Value:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/35722d96-c8f8-4f3d-870d-d4c16e936009)

r2 Value:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/006d6452-35f8-440d-b6f6-3635953fe397)

Data Prediction:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143114486/1d21b78c-ad90-4a9b-9375-e34a6c62c86a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
