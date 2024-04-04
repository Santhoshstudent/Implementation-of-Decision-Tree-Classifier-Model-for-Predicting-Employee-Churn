# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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

![Screenshot 2024-04-04 182708 ml 00001](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/c50ab1c6-c694-481a-8460-ffb6e2e8151e)


![Screenshot 2024-04-04 182948 00002](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/6dd2b2d9-f61f-499a-96fb-236ff0af2922)


![Screenshot 2024-04-04 183155 ml 00003](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/42fa35a0-a980-41e6-8e1a-f46a4ec5a3cd)


![Screenshot 2024-04-04 183319 ml 00005](https://github.com/AkilaMohan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145446853/e039d7a3-18db-448f-8d2e-87b028b7e817)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
