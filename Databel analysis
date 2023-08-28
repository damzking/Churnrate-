import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()
mmscaler = MinMaxScaler()

#Data Cleaning and Organising
df = pd.read_csv('databel.csv')
print(df.head())
print(df['Customer ID'].nunique)
print(df['Customer ID'].value_counts)
print(df.describe())
print(df.columns)


df['new_column'] = np.where(df['Senior'] == 'Yes', 'senior', 
                            np.where(df['Under 30'] == 'Yes', 'under 30', 'other'))
df['demographics'] = df['new_column']
print(df['demographics'].value_counts(normalize=True))
bins = [0, 19, 29, 39, 49, 59, 69, 100]
labels = ['0-19', '20s', '30s', '40s', '50s', '60s', '70+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

df['age_bin'] = pd.cut(df['Age'], bins=range(19, 85, 5), include_lowest=True)
print(df['age_bin'].head())

Local_calls=df['Local Calls']
reshaped_local_calls=np.array(Local_calls).reshape(-1,1)
localcalls_scaled=scaler.fit_transform(reshaped_local_calls)

print(df.demographics.head())
Local_Mins=df['Local Mins']
reshaped_local_Mins=np.array(Local_Mins).reshape(-1,1)
localmins_scaled=scaler.fit_transform(reshaped_local_Mins)

Avg_monthly_Gb=df['Avg Monthly GB Download']
Avg_reshaped=np.array(Avg_monthly_Gb).reshape(-1,1)
Avg_scaled=scaler.fit_transform(Avg_reshaped)

df['Local Calls'] = localcalls_scaled
df['Local Mins'] =  localmins_scaled
df['Avg Monthly GB Download'] = Avg_scaled
#df.to_csv('modified3.csv')
import pandas as pd

# Select the columns to encode
cols = ['Senior','Intl Active', 'Intl Plan', 'Unlimited Data Plan', 'Under 30', 'Group', 'Device Protection & Online Backup','Gender','Contract Type', 'Payment Method','State','demographics']

# Drop rows with missing values
df = df[cols].dropna()

# Loop through the columns and encode them as integers
for col in cols[:7]:
    df[col] = pd.Categorical(df[col]).codes
for col in cols[:5]:
    df = pd.get_dummies(df, columns=[col], prefix = [col])
# Select the columns to one-hot encode


# Drop rows with missing values


# Print the head of the DataFrame to check the results
print(df.head())

df2 = pd.read_csv("modified2.csv", index_col=0).reset_index()
print(df2.head())
col_length = len(df2.columns)

#Y is the target column, X has the rest
X = df2.iloc[:, 2:col_length]
X=pd.get_dummies(df2,columns=['State','demographics'],prefix='ohe',)
X_dup = X
X = X.drop(['Churn Label','Senior','Customer ID','Phone Number','Churn Reason','Churn Category','age_bin','Age (bins)','Age Group'], axis=1) # axis=1 means drop columns, not rows
print(X.columns)
y = df2['Churn Label']
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=99)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
knn_model = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
knn_predictions_train = knn_model.predict(X_train)
knn_predictions_test = knn_model.predict(X_test)
knn_predictions_val = knn_model.predict(X_val)
results_knn = cross_val_score(knn_model,X_train,y_train,cv=kfold,scoring='accuracy')
accuracy_val = accuracy_score(y_val, knn_predictions_val)
accuracy_test = accuracy_score(y_test, knn_predictions_test)
accuracy_train = accuracy_score(y_train, knn_predictions_train)

print("Validation Accuracy with Knn:", accuracy_val)
print("Test Accuracy with Knn:", accuracy_test)
print("Train Accuracy with Knn:", accuracy_train)


classification_report_test = classification_report(y_test, knn_predictions_test)
classification_report_train = classification_report(y_train, knn_predictions_train)
classification_report_val = classification_report(y_val, knn_predictions_val)


cart_model = DecisionTreeClassifier().fit(X_train, y_train) 
cart_predictions_train = cart_model.predict(X_train) 
cart_predictions_test = cart_model.predict(X_test)
cart_predictions_val = cart_model.predict(X_val) 
#print(classification_report(y_train,cart_predictions_train))
#print(classification_report(y_test,cart_predictions_test))
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
results = cross_val_score(cart_model, X_train, y_train, cv=kfold, scoring='accuracy')
#print(results)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
cm = confusion_matrix(y_val, cart_predictions_val)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(13, 7))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt="d")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Decision Tree Model')
plt.show()

results_dec_tree = cross_val_score(cart_model,X_train,y_train,cv=kfold,scoring='accuracy')
accuracy_val = accuracy_score(y_val, cart_predictions_val)
accuracy_test = accuracy_score(y_test, cart_predictions_test)
accuracy_train = accuracy_score(y_train, cart_predictions_train)

print("Validation Accuracy with dec_tree:", accuracy_val)
print("Test Accuracy with dec_tree:", accuracy_test)
print("Train Accuracy with dec_tree:", accuracy_train)


classification_report_test = classification_report(y_test, cart_predictions_test)
classification_report_train = classification_report(y_train, cart_predictions_train)
classification_report_val = classification_report(y_val, cart_predictions_val)

print(classification_report_val)
print(classification_report_train)
print(classification_report_test)

clf = DecisionTreeClassifier(random_state=42)
param_grid = {
   'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}


# Create a GridSearchCV object to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5)
# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)
best_decision_param=grid_search.best_estimator_
y_pred_test_decision_param = best_decision_param.predict(X_test)
y_pred_val_decision_param = best_decision_param.predict(X_val)
y_pred_train_decision_param = best_decision_param.predict(X_train)
# Print the best hyperparameters and their corresponding score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

accuracy_val = accuracy_score(y_val, y_pred_val_decision_param)
accuracy_test = accuracy_score(y_test, y_pred_test_decision_param)
accuracy_train = accuracy_score(y_train, y_pred_train_decision_param)
print("Validation Accuracy with decision tree  h:", accuracy_val)
print("Test Accuracy with decision tree h:", accuracy_test)
print("Train Accuracy with decision tree h:", accuracy_train)
classification_report_test = classification_report(y_test, y_pred_test_decision_param)
classification_report_train = classification_report(y_train,y_pred_train_decision_param)
classification_report_val = classification_report(y_val,y_pred_val_decision_param)
print("Classification Report (Validation):")
print(classification_report_val)
print("Classification Report (Test):")
print(classification_report_test)


cm = confusion_matrix(y_val, y_pred_val_decision_param)
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - decisision trees  (Validation)')
#plt.show()



#now knn with hyperparameter tuning :

#Hyperparameter tuning on knn,logistic regression and decisiontree

# define KNN classifier
knn = KNeighborsClassifier()

# define parameter grid to search
param_grid = {
   'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]}
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# print best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
best_knn = grid_search.best_estimator_
print("best Score",grid_search.best_score_)

#y_pred_train_knn = best_knn.predict(X_train)
best_test_knn =  best_knn.predict(X_test)
best_val_knn = best_knn.predict(X_val)
best_train_knn = best_knn.predict(X_train)


cm = confusion_matrix(y_val, best_val_knn)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(13, 7))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt="d")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - KNN Model')
plt.show()

from sklearn.metrics import accuracy_score

accuracy_val = accuracy_score(y_val,best_val_knn)
#accuracy_train = accuracy_score(y_val,y_pred_train_knn)
accuracy_test = accuracy_score(y_val,best_test_knn) 
print("Test Accuracy:", accuracy_test)
print("Train Accuracy",accuracy_train)
print("validation Accuracy",accuracy_val)
classification_report(y_test, best_test_knn)
classification_report(y_val, best_val_knn)
classification_report(y_train, best_train_knn)



