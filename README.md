# IMPLEMENTATION-OF-LOGISTIC-REGRESSION-MODEL-TO-PREDICT-THE-PLACEMENT-STATUS-OF-STUDENT

## AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## EQUIPMENTS REQUIRED :

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITMM :

1. Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and.duplicated() function respectively

3. Import LabelEncoder and encode the dataset.

4. mport LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array. 6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. Apply new unknown values

## PROGRAM :

```

/*
IMPLEMENTATION-OF-LOGISTIC-REGRESSION-MODEL-TO-PREDICT-THE-PLACEMENT-STATUS-OF-STUDENT

DEVELOPED BY : ANTONY ABISHEK
 
REGISTER NUMBER : 212223240009
*/

```

```import pandas as pd
df=pd.read_csv("Placement_Data.csv")
print(df.head())

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
print(df1.head())

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
print(df1)

x=df1.iloc[:,:-1]
print(x)

y=df1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## OUTPUT :

## ORIGINAL DATA :

<img width="1066" height="206" alt="image" src="https://github.com/user-attachments/assets/b736ea27-d31c-40fb-9735-a46242e3a0f2" />

## AFTER REMOVING :

<img width="832" height="206" alt="image" src="https://github.com/user-attachments/assets/0381186b-4f5e-4cf6-ac95-a5bca22d99eb" />

## NULL DATA :

<img width="564" height="432" alt="image" src="https://github.com/user-attachments/assets/066f47ba-d1fb-42ab-bf5e-046bf7486ef8" />

## LABEL ENCODER :

<img width="970" height="417" alt="image" src="https://github.com/user-attachments/assets/5492344e-4a06-4793-a13a-c6df12d3472e" />

## X :

<img width="471" height="377" alt="image" src="https://github.com/user-attachments/assets/5955acd6-98f3-4708-8ba4-34bb3b8e1586" />

## Y :

<img width="505" height="334" alt="image" src="https://github.com/user-attachments/assets/2a2d7581-6c41-48d4-bf70-469fa68bc507" />

## Y-PREDICTION :

<img width="927" height="82" alt="image" src="https://github.com/user-attachments/assets/4c7b84ca-5916-4bce-b7fe-6493b03c12b1" />

## ACCURACY :

<img width="227" height="43" alt="image" src="https://github.com/user-attachments/assets/89f6420e-7571-49c7-a2a1-9f9537d391c9" />

## COFUSION :

<img width="137" height="90" alt="image" src="https://github.com/user-attachments/assets/0d14c433-2459-40b0-a342-48c10014b396" />

## CLASSIFICATION :

<img width="1050" height="306" alt="image" src="https://github.com/user-attachments/assets/b4c0bb1e-3fa8-406c-a166-75af4b0d1cd7" />

## RESULT :

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
