from google.colab import drive
drive.mount('/content/drive')
import pandas as pd

#loading the data
train_telugu = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TELUGU_TRAINING_DATA.csv')
train_telugu.head()

#cleaning of the training data
text_telugu, y_train_telugu = train_telugu['TEXT DATA'] , train_telugu['LABEL']
spec_chars = ["!",'"',"#","%","&amp","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–", "\n"]
for char in spec_chars:
  train_telugu["TEXT DATA"] = train_telugu["TEXT DATA"].str.replace(char, ' ')
train_telugu.head(10)

#getting count of labels
text_telugu , y_train_telugu = train_telugu["TEXT DATA"], train_telugu["LABEL"]
list_text_telugu = []
for line in text_telugu:
  list_text_telugu.append(line)
y_train_telugu.value_counts()

#vectorization training data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
Vectorizer = TfidfVectorizer(analyzer = 'char' , ngram_range=(1,3))
x_train_telugu = Vectorizer.fit_transform(list_text_telugu)
feature_name_telugu = Vectorizer.get_feature_names_out()
print(feature_name_telugu[:20])
print(x_train_telugu.toarray())

**SVM MODEL**

#import statements
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


#vectorising development data
dev_telugu = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TELUGU_DEVELOPMENT DATA.csv')
Y_dev_telugu = dev_telugu['LABEL']
X_dev_telugu = dev_telugu['TEXT DATA']
#cleaning of development
spec_chars = ["!",'"',"#","%","&amp","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–", "\n"]
for char in spec_chars:
  X_dev_telugu = X_dev_telugu.str.replace(char, ' ')
xdev_list_telugu = []
for line in X_dev_telugu:
  xdev_list_telugu.append(line)
X_dev_telugu= Vectorizer.transform(xdev_list_telugu)

#accuracy of development data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, f1_score
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(x_train_telugu, y_train_telugu)
y_pred_telugu_svm = svm_classifier.predict(X_dev_telugu)
accuracy_telugu_svm = accuracy_score(Y_dev_telugu, y_pred_telugu_svm)
f1_score_tamil=f1_score(Y_dev_telugu, y_pred_telugu_svm, pos_label = 'stressed', average = 'macro')
print(f"f1-score:{f1_score_tamil}")
classification_rep_telugu_svm = classification_report(Y_dev_telugu, y_pred_telugu_svm)
print(f"Accuracy: {accuracy_telugu_svm}")
print("\nClassification Report:")
print(classification_rep_telugu_svm)

#test data extraction
test_telugu = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TELUGU_DEVELOPMENT DATA.csv')


X_test_telugu = test_telugu['TEXT DATA']
#text data cleaning
spec_chars = ["!",'"',"#","%","&amp","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–", "\n"]
for char in spec_chars:
  X_test_telugu = X_test_telugu.str.replace(char, ' ')
xtest_list_telugu = []
for line in X_test_telugu:
  xtest_list_telugu.append(line)
X_test_telugu = Vectorizer.transform(xtest_list_telugu)

svm_classifier.fit(x_train_telugu, y_train_telugu)
y_pred_telugu_svm_test_1 = svm_classifier.predict(X_test_telugu)
test_telugu['class_label'] = list(y_pred_telugu_svm_test_1)
test_telugu_main = pd.DataFrame()
test_telugu_main['TEXT DATA'] = test_telugu['TEXT DATA']
test_telugu_main['LABEL'] = test_telugu['LABEL']
test_telugu_main.to_csv(r'/content/drive/MyDrive/Colab Notebooks/Wit_HUB_StressIdent_LT-EDI@EACL2024_run1.csv', index = False)

**NAIVE** **BAYES**

#training and accuracy of data
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
x_train_telugu_array=x_train_telugu.toarray()
gnb.fit(x_train_telugu_array,y_train_telugu)
y_pred_telugu_gauss=gnb.predict(X_dev_telugu.toarray())
accuracy_telugu_gauss = accuracy_score(Y_dev_telugu, y_pred_telugu_gauss)
f1_score_tamil_2=f1_score(Y_dev_telugu, y_pred_telugu_gauss, pos_label = 'stressed', average = 'macro')
print(f"f1-score:{f1_score_tamil_2}")
classification_rep_telugu_gauss = classification_report(Y_dev_telugu, y_pred_telugu_gauss)
print(f"Accuracy: {accuracy_telugu_gauss}")
print("\nClassification Report:")
print(classification_rep_telugu_gauss)

#uploading predictions to csv file
X_test_telugu_array=X_test_telugu.toarray()
y_pred_telugu_gauss_test_1=gnb.predict(X_test_telugu_array)
test_telugu['LABEL'] = list(y_pred_telugu_gauss_test_1)
test_telugu_main_2 = pd.DataFrame()
test_telugu_main_2['pid'] = test_telugu['TEXT DATA']
test_telugu_main_2['class_label'] = test_telugu['LABEL']
test_telugu_main_2.to_csv(r'/content/drive/MyDrive/Colab Notebooks/Wit_HUB_StressIdent_LT-EDI@EACL2024_run2.csv', index = False)







**RANDOM FOREST CLASSIFIER**

#accuracy of random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train_telugu, y_train_telugu)
y_pred_telugu_rand= classifier.predict(X_dev_telugu)
accuracy_telugu_rand = accuracy_score(Y_dev_telugu, y_pred_telugu_rand)
f1_score_tamil_3=f1_score(Y_dev_telugu, y_pred_telugu_rand, pos_label = 'stressed', average = 'macro')
print(f"f1-score:{f1_score_tamil_3}")
classification_rep_telugu_rand = classification_report(Y_dev_telugu, y_pred_telugu_rand)
print(f"Accuracy: {accuracy_telugu_rand}")
print("\nClassification Report:")
print(classification_rep_telugu_rand)

#uploading predictions to csv file
y_pred_telugu_forest_test_1=classifier.predict(X_test_telugu)
test_telugu['LABEL'] = list(y_pred_telugu_forest_test_1)
test_telugu_main_3 = pd.DataFrame()
test_telugu_main_3['pid'] = test_telugu['TEXT DATA']
test_telugu_main_3['class_label'] = test_telugu['LABEL']
test_telugu_main_3.to_csv(r'/content/drive/MyDrive/Colab Notebooks/Wit_HUB_StressIdent_LT-EDI@EACL2024_run3.csv', index = False)

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# Apply Dimensionality Reduction using PCA
pca = PCA(n_components=100)  # Choose the right number of components based on explained variance
x_train_pca = pca.fit_transform(x_train_telugu.toarray())
X_dev_pca = pca.transform(X_dev_telugu.toarray())

# Hyperparameter tuning for SVM
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='f1_macro', n_jobs=-1)
svm_grid.fit(x_train_pca, y_train_telugu)

# Train optimized SVM model
svm_best = svm_grid.best_estimator_
y_pred_svm_opt = svm_best.predict(X_dev_pca)
print(f"Optimized SVM Accuracy: {accuracy_score(Y_dev_telugu, y_pred_svm_opt)}")
print(f"Optimized SVM F1-score: {f1_score(Y_dev_telugu, y_pred_svm_opt, average='macro')}")
print("\nClassification Report:")
print(classification_report(Y_dev_telugu, y_pred_svm_opt))

# Hyperparameter tuning for Random Forest
rf_params = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None]}
rf_grid = RandomizedSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='f1_macro', n_jobs=-1)
rf_grid.fit(x_train_pca, y_train_telugu)

# Train optimized Random Forest model
rf_best = rf_grid.best_estimator_
y_pred_rf_opt = rf_best.predict(X_dev_pca)
print(f"Optimized RF Accuracy: {accuracy_score(Y_dev_telugu, y_pred_rf_opt)}")
print(f"Optimized RF F1-score: {f1_score(Y_dev_telugu, y_pred_rf_opt, average='macro')}")
print("\nClassification Report:")
print(classification_report(Y_dev_telugu, y_pred_rf_opt))

# Compare pre- and post-optimization results in a table
comparison_table = {
    "Model": ["SVM", "Random Forest"],
    "Before Tuning F1": [f1_score_tamil, f1_score_tamil_3],
    "After Tuning F1": [
        f1_score(Y_dev_telugu, y_pred_svm_opt, average='macro'),
        f1_score(Y_dev_telugu, y_pred_rf_opt, average='macro'),
    ],
    "Before Tuning Accuracy": [accuracy_telugu_svm, accuracy_telugu_rand],
    "After Tuning Accuracy": [
        accuracy_score(Y_dev_telugu, y_pred_svm_opt),
        accuracy_score(Y_dev_telugu, y_pred_rf_opt),
    ],
}

import pandas as pd
print(pd.DataFrame(comparison_table))


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert categorical labels to numerical values
y_train_telugu = label_encoder.fit_transform(y_train_telugu)  # 'Non stressed' -> 0, 'stressed' -> 1
Y_dev_telugu = label_encoder.transform(Y_dev_telugu)  # Apply the same transformation to dev data

# Now train XGBoost
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1)
xgb_clf.fit(x_train_telugu, y_train_telugu)

# Predict on development data
y_pred_xgb = xgb_clf.predict(X_dev_telugu)

# Convert predictions back to original labels if needed
y_pred_xgb_labels = label_encoder.inverse_transform(y_pred_xgb)

# Evaluate performance
from sklearn.metrics import accuracy_score, classification_report, f1_score

accuracy_xgb = accuracy_score(Y_dev_telugu, y_pred_xgb)
f1_xgb = f1_score(Y_dev_telugu, y_pred_xgb, average='macro')

print(f"XGBoost Accuracy: {accuracy_xgb}")
print(f"XGBoost F1-score: {f1_xgb}")
print("\nClassification Report:")
print(classification_report(Y_dev_telugu, y_pred_xgb))
