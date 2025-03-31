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

import numpy as np

if isinstance(y_train_telugu, pd.DataFrame):  
    y_train_telugu = y_train_telugu.squeeze()  # Convert DataFrame to Series

x_train_telugu = np.array(x_train_telugu)
y_train_telugu = np.array(y_train_telugu)


#vectorization training data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
Vectorizer = TfidfVectorizer(analyzer = 'char' , ngram_range=(1,3))
x_train_telugu = Vectorizer.fit_transform(list_text_telugu)
feature_name_telugu = Vectorizer.get_feature_names_out()
print(feature_name_telugu[:20])
print(x_train_telugu.toarray())

from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(sampling_strategy='auto', random_state=42)
x_train_telugu, y_train_telugu = smote.fit_resample(x_train_telugu, y_train_telugu)

print("After SMOTE:", Counter(y_train_telugu))  # Check new distribution


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

from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline

# Apply Dimensionality Reduction using PCA
pca = PCA(n_components=100)
x_train_pca = pca.fit_transform(x_train_telugu.toarray())
X_dev_pca = pca.transform(X_dev_telugu.toarray())

# Train optimized SVM model using Bagging
svm_bagging = BaggingClassifier(estimator=svm_best, n_estimators=10, random_state=42, n_jobs=-1)
svm_bagging.fit(x_train_pca, y_train_telugu)

# Predict with Bagging SVM
y_pred_svm_bagging = svm_bagging.predict(X_dev_pca)

# Train optimized RF model using AdaBoost
rf_adaboost = AdaBoostClassifier(estimator=rf_best, n_estimators=50, random_state=42)
rf_adaboost.fit(x_train_pca, y_train_telugu)

# Predict with AdaBoost RF
y_pred_rf_adaboost = rf_adaboost.predict(X_dev_pca)



# Print results
print(f"Bagging SVM Accuracy: {accuracy_score(Y_dev_telugu, y_pred_svm_bagging)}")
print(f"Bagging SVM F1-score: {f1_score(Y_dev_telugu, y_pred_svm_bagging, average='macro')}\n")

print(f"AdaBoost RF Accuracy: {accuracy_score(Y_dev_telugu, y_pred_rf_adaboost)}")
print(f"AdaBoost RF F1-score: {f1_score(Y_dev_telugu, y_pred_rf_adaboost, average='macro')}\n")


# Compare ensemble methods in a table
comparison_table = {
    "Model": ["Bagging SVM", "AdaBoost RF"],
    "Accuracy": [
        accuracy_score(Y_dev_telugu, y_pred_svm_bagging),
        accuracy_score(Y_dev_telugu, y_pred_rf_adaboost)
    ],
    "F1-score": [
        f1_score(Y_dev_telugu, y_pred_svm_bagging, average='macro'),
        f1_score(Y_dev_telugu, y_pred_rf_adaboost, average='macro')
    ],
}

import pandas as pd
print(pd.DataFrame(comparison_table))