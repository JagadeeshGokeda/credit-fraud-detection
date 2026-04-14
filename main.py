import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
pd.set_option("display.max_columns",None)
pd.set_option('display.width',None)
df = pd.read_csv("creditcard_2023.csv")
#print(df.head())
x = df.drop(["id","Class"],axis = 1)
y = df["Class"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify=y,random_state=42)
scaler = StandardScaler()
x_train["Amount"] = scaler.fit_transform(x_train[["Amount"]])
x_test["Amount"] = scaler.transform(x_test[["Amount"]])
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(x_train,y_train)
log_reg_pred = log_reg.predict(x_test)
dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
accuracy_log_reg = accuracy_score(y_test,log_reg_pred)
cm = confusion_matrix(y_test,log_reg_pred)
cr = classification_report(y_test,log_reg_pred)
print("     EVALUTION OF LOGISTIC REGRESSION      \n")
print(f"accuracy score is : {accuracy_log_reg}")
print(f"confusion matrix is : \n {cm}")
print(f"classification report is\n {cr}")
accuracy_dt = accuracy_score(y_test,dt_pred)
cm_dt = confusion_matrix(y_test,dt_pred)
cr_dt = classification_report(y_test,dt_pred)
print("     EVALUTION OF DECISION TREE     \n")
print(f"accuracy score is : {accuracy_dt}")
print(f"confusion matrix is : \n {cm_dt}")
print(f"classification report is\n {cr_dt}")
prob_log_reg = log_reg.predict_proba(x_test)[:,1]
prob_dt = dt.predict_proba(x_test)[:,1]
fpr_log_reg,tpr_log_reg,_ = roc_curve(y_test,prob_log_reg)
fpr_dt,tpr_dt,_ = roc_curve(y_test,prob_dt)
auc_log_reg = roc_auc_score(y_test,prob_log_reg)
auc_dt = roc_auc_score(y_test,prob_dt)
print("      AUC SCORES     \n")
print(f"auc score of logisti regression: {auc_log_reg}")
print(f"auc score of decision tree : {auc_dt}")
plt.plot(fpr_log_reg, tpr_log_reg, label=f"Logistic Regression")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree")
plt.plot([0,1], [0,1], linestyle='--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC CURVE COMPARISION")
plt.legend()
plt.show()