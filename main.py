
1)linear
import pandas as pd
import numpy as np

file_path = 'BostonHousing.csv'  # Update this if your file path is different
data = pd.read_csv(file_path)

print(data.head())

# Display information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

data = data.dropna()

# Initial linear regression with all parameters
X_all = data.drop(columns=['medv'])
y_all = data['medv']

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model_all = LinearRegression()

# Train the model
model_all.fit(X_train_all, y_train_all)

y_pred_all = model_all.predict(X_test_all)

mse_all = mean_squared_error(y_test_all, y_pred_all)
rmse_all = np.sqrt(mse_all)
r2_all = r2_score(y_test_all, y_pred_all)

print(f"All Parameters - Mean Squared Error: {mse_all}")
print(f"All Parameters - Root Mean Squared Error: {rmse_all}")
print(f"All Parameters - R-squared: {r2_all}")

# Plotting actual vs predicted values for all parameters
plt.figure(figsize=(10, 6))
plt.scatter(y_test_all, y_pred_all, edgecolor='k', alpha=0.7)
plt.plot([y_test_all.min(), y_test_all.max()], [y_test_all.min(), y_test_all.max()], 'r--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('All Parameters - Actual vs Predicted Values')
plt.show()

# Draw a heatmap of correlations
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

relevant_features = corr_matrix.index[abs(corr_matrix["medv"]) > 0.5].tolist()
relevant_features.remove('medv')
print("Selected relevant features:", relevant_features)

# Linear regression with selected parameters
X_relevant = data[relevant_features]
y_relevant = data['medv']

# Split the data into training and testing sets
X_train_rel, X_test_rel, y_train_rel, y_test_rel = train_test_split(X_relevant, y_relevant, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model_rel = LinearRegression()
1)linear
# Train the model
model_rel.fit(X_train_rel, y_train_rel)

y_pred_rel = model_rel.predict(X_test_rel)

# Evaluate the model
mse_rel = mean_squared_error(y_test_rel, y_pred_rel)
rmse_rel = np.sqrt(mse_rel)
r2_rel = r2_score(y_test_rel, y_pred_rel)

print(f"Selected Parameters - Mean Squared Error: {mse_rel}")
print(f"Selected Parameters - Root Mean Squared Error: {rmse_rel}")
print(f"Selected Parameters - R-squared: {r2_rel}")

# Plotting actual vs predicted values for selected parameters
plt.figure(figsize=(10, 6))
plt.scatter(y_test_rel, y_pred_rel, edgecolor='k', alpha=0.7)
plt.plot([y_test_rel.min(), y_test_rel.max()], [y_test_rel.min(), y_test_rel.max()], 'r--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Selected Parameters - Actual vs Predicted Values')
plt.show()

# Compare results
print("\nComparison:")
print(f"All Parameters - RMSE: {rmse_all}, R-squared: {r2_all}")
print(f"Selected Parameters - RMSE: {rmse_rel}, R-squared: {r2_rel}")


2)logistic
import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())

print(df.info())

df = df[['Survived', 'Age', 'Sex', 'Pclass']]
df = pd.get_dummies(df, columns=['Sex', 'Pclass'])
df.dropna(inplace=True)
print(df.head())

print(df)

from sklearn.model_selection import train_test_split

x = df.drop('Survived', axis=1)
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

model.score(x_test, y_test)

from sklearn.model_selection import cross_val_score

cross_val_score(model, x, y, cv=5).mean()

from sklearn.metrics import confusion_matrix

y_predicted = model.predict(x_test)
confusion_matrix(y_test, y_predicted)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

y_pred = model.predict(x_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Perished', 'Survived'])
disp.plot(cmap='Blues')

# Optional: customize the plot further
plt.xticks(rotation='vertical')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predicted))

accuracy = model.score(x_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

from sklearn.metrics import roc_curve, RocCurveDisplay
y_prob = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Create the ROC curve display
disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
disp.plot()

# Add labels and title if desired
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()


3)descion

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

csv_path = 'adult_dataset.csv'
df = pd.read_csv(csv_path)

print(df.head())

print ("Rows     : \n" ,df.shape[0])
print ("Columns  : \n" ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values : \n", df.isnull().sum().values.sum())
print ("\nUnique values : \n", df.nunique())

df.info()

print(df.describe())

df_missing_workclass = (df['workclass']=='?').sum()
df_missing_workclass

df_missing = (df=='?').sum()
df_missing

percent_missing = (df=='?').sum() * 100/len(df)
percent_missing

df.apply(lambda x: x !='?',axis=1).sum()

df_categorical = df.select_dtypes(include=['object'])

# checking whether any other column contains '?' value
df_categorical.apply(lambda x: x=='?',axis=1).sum()

df = df[df['native.country'] != '?']
df = df[df['occupation'] !='?']

print(df)

df.info()

from sklearn import preprocessing

# encode categorical variables using label Encoder
# select all categorical variables
df_categorical = df.select_dtypes(include=['object'])
print(df_categorical.head())

#appy label encoding
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
print(df_categorical.head())

df = df.drop(df_categorical.columns,axis=1)
print(df)

df = pd.concat([df,df_categorical],axis=1)
print(df.head())

df['income'] = df['income'].astype('category')

print(df)

from sklearn.model_selection import train_test_split

# independent features to X
X = df.drop('income',axis=1)

# dependent variable to Y
Y = df['income']

print(X.head())

Y.head()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=99)

print(X_train.head())

Y_train.head()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

dec_tree.fit(X_train, Y_train)

Y_pred_dec_tree = dec_tree.predict(X_test)
Y_pred_dec_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print('Decision Tree Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_dec_tree) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_dec_tree) * 100, 2))

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_dec_tree)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Reds')

from sklearn import tree
import matplotlib.pyplot as plt

# Assuming 'clf' is your trained decision tree classifier
plt.figure(figsize=(20,10))
tree.plot_tree(dec_tree, filled=True)
plt.show()

from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',  # You can change this to 'f1' if you prefer
                           cv=5,  # 5-fold cross-validation
                           verbose=1,
                           n_jobs=-1)

# Fit the model using GridSearchCV
grid_search.fit(X_train, Y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

best_dec_tree = grid_search.best_estimator_
Y_pred_best_dec_tree = best_dec_tree.predict(X_test)

print('Tuned Decision Tree Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_best_dec_tree) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_best_dec_tree) * 100, 2))

cm_best = confusion_matrix(Y_test, Y_pred_best_dec_tree)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best)
disp_best.plot(cmap='Blues')

plt.figure(figsize=(20,10))
tree.plot_tree(best_dec_tree, max_depth=5, filled=True, fontsize=10)
plt.title('Optimized Decision Tree (Depth = 5)')
plt.show()

"""Before Hyperparameter Tuning

> Add blockquote


"""

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

precision_before = precision_score(Y_test, Y_pred_dec_tree)
recall_before = recall_score(Y_test, Y_pred_dec_tree)
accuracy_before = accuracy_score(Y_test, Y_pred_dec_tree)
f1_before = f1_score(Y_test, Y_pred_dec_tree)
confusion_matrix_before = confusion_matrix(Y_test, Y_pred_dec_tree)

print("Before Tuning")
print(f"Accuracy: {accuracy_before:.2f}")
print(f"F1 Score: {f1_before:.2f}")
print(f"Precision: {precision_before:.2f}")
print(f"Recall: {recall_before:.2f}")
print(f"Confusion Matrix: \n{confusion_matrix_before}")

"""After Hyperparameter Tuning


"""

precision_after = precision_score(Y_test, Y_pred_best_dec_tree)
recall_after = recall_score(Y_test, Y_pred_best_dec_tree)
accuracy_after = accuracy_score(Y_test, Y_pred_best_dec_tree)
f1_after = f1_score(Y_test, Y_pred_best_dec_tree)
confusion_matrix_after = confusion_matrix(Y_test, Y_pred_best_dec_tree)

print("After Tuning")
print(f"Accuracy: {accuracy_after:.2f}")
print(f"F1 Score: {f1_after:.2f}")
print(f"Precision: {precision_after:.2f}")
print(f"Recall: {recall_after:.2f}")
print(f"Confusion Matrix: \n{confusion_matrix_after}")



random
# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

csv_path = 'adult_dataset.csv'
df = pd.read_csv(csv_path)

print(df.head())

print ("Rows     : \n" ,df.shape[0])
print ("Columns  : \n" ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values : \n", df.isnull().sum().values.sum())
print ("\nUnique values : \n", df.nunique())

df.info()

print(df.describe())

df_missing_workclass = (df['workclass']=='?').sum()
df_missing_workclass

df_missing = (df=='?').sum()
df_missing

percent_missing = (df=='?').sum() * 100/len(df)
percent_missing

df.apply(lambda x: x !='?',axis=1).sum()

df_categorical = df.select_dtypes(include=['object'])

# checking whether any other column contains '?' value
df_categorical.apply(lambda x: x=='?',axis=1).sum()

df = df[df['native.country'] != '?']
df = df[df['occupation'] !='?']

print(df)

df.info()

from sklearn import preprocessing

# encode categorical variables using label Encoder
# select all categorical variables
df_categorical = df.select_dtypes(include=['object'])
print(df_categorical.head())

#appy label encoding
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
print(df_categorical.head())

df = df.drop(df_categorical.columns,axis=1)
print(df)

df = pd.concat([df,df_categorical],axis=1)
print(df.head())

df['income'] = df['income'].astype('category')

print(df)

from sklearn.model_selection import train_test_split

# independent features to X
X = df.drop('income',axis=1)

# dependent variable to Y
Y = df['income']

print(X.head())

Y.head()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=99)

print(X_train.head())

Y_train.head()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf.fit(X_train, Y_train)

# Predict the labels on the test data
Y_pred_rf = rf.predict(X_test)

# Evaluate the performance of the model
print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_rf) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_rf) * 100, 2))

# Confusion matrix
cm_rf = confusion_matrix(Y_test, Y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap='Reds')

from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid to search
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

# Create the RandomizedSearchCV object
random_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                      param_distributions=param_grid_rf,
                                      n_iter=20,  # Number of parameter settings that are sampled
                                      scoring='accuracy',
                                      cv=3,  # 3-fold cross-validation
                                      verbose=1,
                                      n_jobs=-1,
                                      random_state=42)

# Fit the model using RandomizedSearchCV
random_search_rf.fit(X_train, Y_train)

# Best parameters and score
print(f"Best Parameters: {random_search_rf.best_params_}")
print(f"Best Score: {random_search_rf.best_score_}")

# Use the best estimator to predict the test set
best_rf = random_search_rf.best_estimator_
Y_pred_best_rf = best_rf.predict(X_test)

print('Tuned Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_best_rf) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_best_rf) * 100, 2))

# Confusion matrix for the tuned model
cm_best_rf = confusion_matrix(Y_test, Y_pred_best_rf)
disp_best_rf = ConfusionMatrixDisplay(confusion_matrix=cm_best_rf)
disp_best_rf.plot(cmap='Blues')

from sklearn import tree

# Plot one of the trees in the Random Forest (for visualization)
plt.figure(figsize=(20, 10))
tree.plot_tree(best_rf.estimators_[0], max_depth=5, filled=True, fontsize=10)
plt.title('Optimized Random Forest Tree (Depth = 5)')
plt.show()

from sklearn.metrics import precision_score, recall_score

# Before tuning
precision_before = precision_score(Y_test, Y_pred_rf)
recall_before = recall_score(Y_test, Y_pred_rf)
accuracy_before = accuracy_score(Y_test, Y_pred_rf)
f1_before = f1_score(Y_test, Y_pred_rf)
confusion_matrix_before = confusion_matrix(Y_test, Y_pred_rf)

print("Before Tuning")
print(f"Accuracy: {accuracy_before:.2f}")
print(f"F1 Score: {f1_before:.2f}")
print(f"Precision: {precision_before:.2f}")
print(f"Recall: {recall_before:.2f}")
print(f"Confusion Matrix: \n{confusion_matrix_before}")

# After tuning
precision_after = precision_score(Y_test, Y_pred_best_rf)
recall_after = recall_score(Y_test, Y_pred_best_rf)
accuracy_after = accuracy_score(Y_test, Y_pred_best_rf)
f1_after = f1_score(Y_test, Y_pred_best_rf)
confusion_matrix_after = confusion_matrix(Y_test, Y_pred_best_rf)

print("After Tuning")
print(f"Accuracy: {accuracy_after:.2f}")
print(f"F1 Score: {f1_after:.2f}")
print(f"Precision: {precision_after:.2f}")
print(f"Recall: {recall_after:.2f}")
print(f"Confusion Matrix: \n{confusion_matrix_after}")


bagging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Load and preprocess the dataset
df = pd.read_csv('adult_dataset.csv')
df.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

df['income'] = LabelEncoder().fit_transform(df['income'])

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train AdaBoost
ada_classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)

# Initialize and train Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)

# Initialize and train XGBoost
xgb_classifier = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

# Initialize and train LightGBM
lgb_classifier = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgb_classifier.fit(X_train, y_train)
y_pred_lgb = lgb_classifier.predict(X_test)

# Initialize and train CatBoost
catboost_classifier = CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=3, random_state=42, verbose=0)
catboost_classifier.fit(X_train, y_train)
y_pred_catboost = catboost_classifier.predict(X_test)

# Compare performance
def print_comparison(name, y_true, y_pred):
    print(f"{name}")
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.2f}')
    print(classification_report(y_true, y_pred))
    print("-" * 50)

print_comparison("AdaBoost", y_test, y_pred_ada)
print_comparison("Gradient Boosting", y_test, y_pred_gb)
print_comparison("XGBoost", y_test, y_pred_xgb)
print_comparison("LightGBM", y_test, y_pred_lgb)
print_comparison("CatBoost", y_test, y_pred_catboost)



hierachal
# Commented out IPython magic to ensure Python compatibility.
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
# %matplotlib inline

data=pd.read_csv("Wholesale customers data.csv")
print(data.head())

#normalize the data
scaled=normalize(data)
scaled=pd.DataFrame(scaled,columns=data.columns)
print(scaled.head())

#dendrogram to determine the number of clusters
#x axis: samples ; y axis: distance between samples
plt.figure(figsize=(10,7))
plt.title("Dendrogram")
Z=linkage(scaled,method='ward')
dendrograms=dendrogram(Z)

#threshold
#from the dendrogram we choose y=6 as the threshold
plt.figure(figsize=(10,7))
plt.title("Dendrogram")
Z=linkage(scaled,method='ward')
dendrograms=dendrogram(Z)
plt.axhline(y=6,color='black')

cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
cluster.fit_predict(scaled)
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='euclidean')
cluster.fit_predict(scaled)

#visualization
plt.figure(figsize=(10,7))
plt.scatter(scaled['Milk'],scaled['Grocery'],c=cluster.labels_,cmap = mcolors.ListedColormap(["yellow", "green"]))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Load the dataset
file_path = '/content/Wholesale customers data.csv'
wholesale_data = pd.read_csv(file_path)

# Select the spending columns for clustering
spending_data = wholesale_data.iloc[:, 2:]

# Normalize the data
scaler = StandardScaler()
spending_data_normalized = scaler.fit_transform(spending_data)

# Plot the dendrogram to find the optimal number of clusters
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(spending_data_normalized, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()


agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = agg_clustering.fit_predict(spending_data_normalized)

# Calculate the silhouette score to evaluate clustering
silhouette_avg = silhouette_score(spending_data_normalized, cluster_labels)

# Output the silhouette score
silhouette_avg



dimensionreduct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
url = "adult_dataset.csv"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=columns, na_values=' ?')

data.dropna(inplace=True)

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

X = data.drop('income', axis=1)
y = data['income']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

accuracy_scores = []

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

accuracy_without_pca = accuracy_score(y_test, y_pred)
accuracy_scores.append(accuracy_without_pca)

print("Logistic Regression without PCA")
print(f"Accuracy: {accuracy_without_pca:.4f}")
print(classification_report(y_test, y_pred))

pca_all = PCA()
X_train_pca_all = pca_all.fit_transform(X_train)
X_test_pca_all = pca_all.transform(X_test)

log_reg_pca_all = LogisticRegression()
log_reg_pca_all.fit(X_train_pca_all, y_train)
y_pred_pca_all = log_reg_pca_all.predict(X_test_pca_all)

accuracy_pca_all = accuracy_score(y_test, y_pred_pca_all)
accuracy_scores.append(accuracy_pca_all)

print("\nLogistic Regression with PCA (whole dataset)")
print(f"Accuracy: {accuracy_pca_all:.4f}")
print(classification_report(y_test, y_pred_pca_all))

pca_50 = PCA(0.5)  # Retain components that explain 50% of the variance
X_train_pca_50 = pca_50.fit_transform(X_train)
X_test_pca_50 = pca_50.transform(X_test)

log_reg_pca_50 = LogisticRegression()
log_reg_pca_50.fit(X_train_pca_50, y_train)
y_pred_pca_50 = log_reg_pca_50.predict(X_test_pca_50)

accuracy_pca_50 = accuracy_score(y_test, y_pred_pca_50)
accuracy_scores.append(accuracy_pca_50)

print("\nLogistic Regression with PCA (variance explained ≥ 0.5)")
print(f"Accuracy: {accuracy_pca_50:.4f}")
print(classification_report(y_test, y_pred_pca_50))

pca_75 = PCA(0.75)  # Retain components that explain 75% of the variance
X_train_pca_75 = pca_75.fit_transform(X_train)
X_test_pca_75 = pca_75.transform(X_test)

log_reg_pca_75 = LogisticRegression()
log_reg_pca_75.fit(X_train_pca_75, y_train)
y_pred_pca_75 = log_reg_pca_75.predict(X_test_pca_75)

accuracy_pca_75 = accuracy_score(y_test, y_pred_pca_75)
accuracy_scores.append(accuracy_pca_75)

print("\nLogistic Regression with PCA (variance explained ≥ 0.75)")
print(f"Accuracy: {accuracy_pca_75:.4f}")
print(classification_report(y_test, y_pred_pca_75))

# Plotting Explained Variance for Each Scenario
plt.figure(figsize=(10, 6))
explained_variance = np.cumsum(pca_all.explained_variance_ratio_)
plt.plot(explained_variance, marker='o', linestyle='--', label='Cumulative Explained Variance (PCA All)')
plt.axhline(y=0.5, color='r', linestyle='--', label='0.5 Variance Threshold')
plt.axhline(y=0.75, color='g', linestyle='--', label='0.75 Variance Threshold')
plt.title('Explained Variance by Number of PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Accuracy Comparison Graph
plt.figure(figsize=(8, 5))
scenarios = ['Without PCA', 'PCA All Components', 'PCA ≥ 0.5', 'PCA ≥ 0.75']
plt.bar(scenarios, accuracy_scores, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Accuracy Across Different PCA Scenarios')
plt.xlabel('Scenario')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.0)
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=12)
plt.show()

# Confusion Matrix for the Best Model (You can choose the best model here)
best_model_conf_matrix = confusion_matrix(y_test, y_pred_pca_75)

plt.figure(figsize=(6, 4))
sns.heatmap(best_model_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Best Model (PCA ≥ 0.75)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



























































































































































import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import google.generativeai as genai
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Configure the Google Gemini model with API key
api_key = os.getenv('GENAI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

# Function to fetch stock data from Yahoo Finance using yfinance
def fetch_yfinance_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")  # Fetching 1 year of historical data
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol} from Yahoo Finance: {str(e)}")
        return None

# Function to interact with Gemini AI for stock-related queries
def get_gemini_response(symbol):
    prompt_template = (
        ""You are an intelligent assistant with expertise in stock market analysis. Provide a comprehensive analysis of the following stock, including performance trends, financial ratios, and outlook based on recent news. Also, assess the company's dividend history and payout consistency, comparing it with other similar stocks to determine if it offers a better dividend yield."

"Stock: {}"

\n"
        "Analysis:"
    )

    response = model.generate_content(prompt_template.format(symbol), stream=True)
    full_text = ""
    for chunk in response:
        full_text += chunk.text
    return full_text

# Streamlit app setup
st.set_page_config(page_title="Stock Data Dashboard", layout="wide")
st.title("Stock Data Dashboard")

# Streamlit inputs
symbols = st.multiselect("Select Stock Symbols", ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO", 
    "HINDUNILVR.BO", "BHARTIARTL.BO", "ITC.BO", "KOTAKBANK.BO", "SBI.BO",
    "LTI.BO", "WIPRO.BO", "HCLTECH.BO", "M&M.BO", "ADANIGREEN.BO", "NTPC.BO",
    "POWERGRID.BO", "ONGC.BO", "BAJFINANCE.BO", "JSWSTEEL.BO", "HDFC.BO",
    "M&MFIN.BO", "SBILIFE.BO", "CIPLA.BO", "DRREDDY.BO", "SUNPHARMA.BO", "TSLA"], default=["RELIANCE.BO"])

if st.button("Fetch Data"):
    if symbols:
        with st.spinner("Fetching stock data..."):  # Loading spinner
            all_figures = []
            all_data = {}  # To store data for CSV download
            for symbol in symbols:
                data = fetch_yfinance_stock_data(symbol)
                if data is not None and not data.empty:
                    all_data[symbol] = data  # Store data for download
                    st.subheader(f"Stock Data for {symbol}")

                    # Display raw data
                    st.write("**Displaying last 5 records:**")
                    st.write(data.tail())

                    # Create a column layout for visualizations
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Stock Price Chart with Trendline
                        st.subheader(f"Stock Price Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Close'].plot(ax=ax, title=f"Closing Prices for {symbol}", legend=True)

                        # Adding a trendline
                        x = np.arange(len(data))
                        z = np.polyfit(x, data['Close'], 1)  # Linear fit
                        p = np.poly1d(z)
                        ax.plot(data.index, p(x), color='red', linestyle='--', label='Trendline')

                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        st.pyplot(fig)
                        all_figures.append(fig)

                    with col2:
                        # Volume Traded Chart
                        st.subheader(f"Volume Traded Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Volume'].plot(ax=ax, color='orange', title=f"Volume Traded for {symbol}", legend=True)
                        plt.xlabel("Date")
                        plt.ylabel("Volume")
                        st.pyplot(fig)
                        all_figures.append(fig)

                    with col3:
                        # Moving Average Chart
                        st.subheader(f"Moving Average Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Close'].rolling(window=30).mean().plot(ax=ax, color='blue', label='30-Day Moving Average')
                        data['Close'].plot(ax=ax, title=f"Closing Prices and Moving Average for {symbol}", legend=True)
                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        st.pyplot(fig)
                        all_figures.append(fig)

                    # Fetch and display Gemini response
                    gemini_response = get_gemini_response(symbol)
                    st.subheader(f"Gemini Response for {symbol}")
                    st.write(gemini_response)

                    # Future Trend Analysis
                    st.subheader(f"Future Trend Analysis for {symbol}")
                    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                    future_prices = data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.05, size=30)).cumprod()  # Simulated prices
                    future_data = pd.Series(future_prices, index=future_dates)

                    # Plot future trend
                    fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
                    data['Close'].plot(ax=ax, label='Historical Prices', legend=True, fontsize=8)
                    future_data.plot(ax=ax, label='Projected Future Prices', color='green', linestyle='--', legend=True, fontsize=8)

                    plt.title(f"Projected Future Prices for {symbol}", fontsize=10)
                    plt.xlabel("Date", fontsize=8)
                    plt.ylabel("Price (USD)", fontsize=8)
                    plt.legend(fontsize=8)
                    st.pyplot(fig)
                    all_figures.append(fig)

                    # CSV Download
                    csv = data.to_csv().encode()
                    st.download_button(f"Download {symbol} Data as CSV", csv, f"{symbol}_data.csv", "text/csv")

                else:
                    st.error(f"Failed to fetch data for {symbol}")

    else:
        st.error("Please select at least one stock symbol.")
