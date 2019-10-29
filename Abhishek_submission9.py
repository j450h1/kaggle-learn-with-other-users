    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


#Abhishek 10/28/2019, added low_memor parameter due to Memory error in concatenate chunks  and got errors
# Zipped the files manaully and changed the file name to .zip in test and train dataset population

test = pd.read_csv(Path.cwd() / "test.zip")
# strange for me its test.csv.zip
train = pd.read_csv(Path.cwd() / "train.zip")
# strange for me its train.csv.zip

train.head()


# Create local validation
validation_train, validation_test = train_test_split(train, test_size=0.3, random_state=123)

features_selected = ['Elevation', 'Wilderness_Area4', 'Horizontal_Distance_To_Roadways', 'Soil_Type10', 'Soil_Type3',
'Wilderness_Area1','Soil_Type38','Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Hydrology','Soil_Type39'] #x
target = ['Cover_Type'] #y
X = train[features_selected]
y = train[target]
X_train = validation_train[features_selected]
y_train = validation_train[target]
X_test = validation_test[features_selected]
y_test = validation_test[target] #switch to all 4 in one as shown elsewhere

from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
model = RandomForestClassifier(n_estimators=100, max_depth=35, random_state=0)
model.fit(X_train, y_train)  


import pandas as pd
feature_imp = pd.Series(model.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp


# Visualize

import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# prediction on test set
y_pred = model.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


submission = test.copy()

submission['Cover_Type'] = model.predict(submission[features_selected])

#submission.info(memory_usage='deep')

submission[['Id','Cover_Type']].to_csv('submission{0}.csv'.format(9), index=False)

