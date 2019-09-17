# -*- coding: utf-8 -*-

###############################################################################################################################
# Clasificadores de La Nación Noticias
###############################################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

SEED = 15
K_FOLD = 5
JOBS = 4

# Leemos el dataset
dataset = pd.read_pickle(r'la_nacion\datasets\la_nacion_dataset.pkl')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# clases a números
le = LabelEncoder()
y = le.fit_transform(y)
# números y categorias
num_clase = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

pipe = Pipeline([('scl', SelectKBest(score_func=chi2)),
                 ('clf', RandomForestClassifier(random_state=SEED))])

# search_space = [{'clf': [LogisticRegression(random_state=SEED)],
#                  'clf__penalty': ['l1', 'l2'],
#                  'clf__C': np.logspace(-5, 5, 20),
#                  'clf__solver': ['liblinear'],
#                  'clf__multi_class': ['auto']},
#                 {'clf': [RandomForestClassifier(random_state=SEED)],
#                  'clf__criterion': ['gini', 'entropy'],
#                  'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                  'clf__max_depth': range(3, 30, 2),
#                  'clf__min_samples_split': [2, 5, 10],
#                  'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
#                 {'clf': [SVC(random_state=SEED)],
#                  'clf__kernel': ['linear'],
#                  'clf__C': np.logspace(-5, 5, 20)}]

search_space = [{'clf': [LogisticRegression(random_state=SEED)],
                 'scl__k':[100, 150, 200],
                 'clf__C': np.logspace(-5, 5, 1),
                 'clf__solver': ['liblinear'],
                 'clf__multi_class': ['auto']},
                {'clf': [RandomForestClassifier(random_state=SEED)],
                 'scl__k': [100, 150, 200],
                 'clf__min_samples_leaf': [1, 2],
                 'clf__n_estimators': [4, 8]},
                {'clf': [SVC(random_state=SEED)],
                 'scl__k': [100, 150, 200],
                 'clf__kernel': ['linear'],
                 'clf__C': np.logspace(-5, 5, 1)}]

gs = GridSearchCV(estimator=pipe,
                  param_grid=search_space,
                  scoring='accuracy',
                  cv=K_FOLD,
                  n_jobs=JOBS,
                  verbose=1)

gs.fit(X_train, y_train)

# FutureWarning: You are accessing a training score ('split3_train_score'), which will not be available by default any more in 0.21. If you need training scores, please set return_train_score=True
cv_result = pd.DataFrame(gs.cv_results_)

final_features_idx = gs.best_estimator_.named_steps["scl"].get_support(indices=True)
feature_names = X_train.columns[final_features_idx]

gs.predict(X)
