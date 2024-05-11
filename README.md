# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
    

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: santhosh kumar B

RegisterNumber: 212223230193

*/
```
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree

data=pd.read_csv("Employee_EX6.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])

data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y=data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

plt.figure(figsize=(18,6))

plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)

plt.show()


## Output:
![decision tree classifier model](sam.png)

![Screenshot 2024-04-04 182708 ml 00001](https://github.com/Santhoshstudent/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/555b63c0-3f97-47c0-a4a0-b86dee562ac3)

![Screenshot 2024-04-04 182948 00002](https://github.com/Santhoshstudent/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/0195c6da-6031-4bc2-a668-66a474ce5076)

![Screenshot 2024-04-04 183155 ml 00003](https://github.com/Santhoshstudent/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/8b8d29df-ea76-40cb-a404-28d4196b92f3)

![Screenshot 2024-04-04 183319 ml 00005](https://github.com/Santhoshstudent/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/23b7dc30-d448-4020-9e51-b53df9828dcc)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
