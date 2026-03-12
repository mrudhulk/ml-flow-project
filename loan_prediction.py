
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
from mlflow import set_experiment, start_run, set_tag
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

# Load the dataset
dataset = pd.read_csv("train.csv")
numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = dataset.select_dtypes(include=[object]).columns.tolist()
categorical_columns.remove("Loan_Status")
categorical_columns.remove("Loan_ID")

# Handle missing values
for col in numeric_columns:
    #dataset[col].fillna(dataset[col].median(), inplace=True)
     dataset.fillna({col: dataset[col].mean()}, inplace=True)
    
for col in categorical_columns:
    #dataset[col].fillna(dataset[col].mode()[0], inplace=True)
    dataset.fillna({col: dataset[col].mode()[0]}, inplace=True)
# Take care of outliers in the dataset

dataset[numeric_columns] = dataset[numeric_columns].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Log Transforamtion & Domain Processing
dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome']).copy()


# Dropping ApplicantIncome and CoapplicantIncome
dataset = dataset.drop(columns=['ApplicantIncome','CoapplicantIncome'])

# Label encoding categorical variables
for col in categorical_columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

#Encode the target columns
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])

# Train test split
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset.Loan_Status
RANDOM_SEED = 6

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)

# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_forest, cv=5, n_jobs=-1)
model_rf = grid_rf.fit(X_train, y_train)

#Logistic Regression
lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, n_jobs=-1)
model_lr = grid_lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
param_grid_dt = {
    'max_depth': [3, 5, 7, 9, 11, 13],
    'criterion': ['gini', 'entropy'],
}
grid_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
model_dt = grid_dt.fit(X_train, y_train)

mlflow.set_experiment("Loan_Prediction_Experiment")

#Model Evaluation metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc)

def mlflow_log_model(model, X, y, model_name):
    with mlflow.start_run(run_name=model_name):
    #with mlflow.start_run() as run:
        #mlflow.set_tracking_uri("http://0.0.0.0:5001/")
        #run_id = run.info.run_id
        set_tag("version","1.0.0")
        log_param("model_type", model_name)
        y_pred = model.predict(X)
        accuracy, f1, auc = eval_metrics(y, y_pred)
        log_metric("accuracy", accuracy)
        log_metric("f1_score", f1)
        log_metric("auc", auc)

        mlflow.log_artifact("plots/ROC_curve.png")
        #mlflow.log_artifact("plots/ROC_curve.png", name="plots")
        
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} logged successfully!")


mlflow_log_model(model_rf, X_test, y_test, "RandomForest")
mlflow_log_model(model_lr, X_test, y_test, "LogisticRegression")
mlflow_log_model(model_dt, X_test, y_test, "DecisionTree")