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
MAX_FEATURES = 150
K_FOLD = 5
JOBS = -1
N_ITER = 500

# Leemos el dataset
dataset = pd.read_pickle(r'la_nacion\datasets\la_nacion_dataset.pkl')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# clases a números
le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(y), columns=['target'])
# números y categorias
num_clase = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# pipelines
pipe_lr = Pipeline([('scl', SelectKBest(score_func=chi2, k=MAX_FEATURES)),
                    ('clf', LogisticRegression(random_state=SEED))])

pipe_svc = Pipeline([('scl', SelectKBest(score_func=chi2, k=MAX_FEATURES)),
                     ('clf', SVC(random_state=SEED))])

pipe_rf = Pipeline([('scl', SelectKBest(score_func=chi2, k=MAX_FEATURES)),
                    ('clf', RandomForestClassifier(random_state=SEED))])

# grilla de parámetros
grid_params_lr = {'clf__penalty': ['l1', 'l2'],
                  'clf__C': np.logspace(-5, 5, 20),
                  'clf__solver': ['liblinear']}

grid_params_rf = {'clf__criterion': ['gini', 'entropy'],
                  'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'clf__max_depth': range(3, 30, 2),
                  'clf__min_samples_split': [2, 5, 10],
                  'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]}

grid_params_svc = {'clf__kernel': ['linear'],
                   'clf__C': np.logspace(-5, 5, 20)}

# busquedas
gs_lr = GridSearchCV(estimator=pipe_lr,
                     param_grid=grid_params_lr,
                     scoring='accuracy',
                     cv=K_FOLD,
                     n_jobs=JOBS)

gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=grid_params_rf,
                     scoring='accuracy',
                     cv=K_FOLD,
                     n_jobs=JOBS)

gs_svc = GridSearchCV(estimator=pipe_svc,
                      param_grid=grid_params_svc,
                      scoring='accuracy',
                      cv=K_FOLD,
                      n_jobs=JOBS)

# lista de pipelines
grids = [gs_lr, gs_rf, gs_svc]

# diccionario de pipelines y clasificadores
grid_dict = {0: 'Logistic Regression', 1: 'Random Forest',
             2: 'Support Vector Classifier'}

# Ajuste
print('Optimizando modelos...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimador: %s' % grid_dict[idx])
    # ajusta búsqueda
    gs.fit(X_train, y_train)
    # mejores parámetros
    print('Mejores parámetros: %s' % gs.best_params_)
    # Mejor accuracy en entrenamiento
    print('Mejor accuracy en entrenamiento: %.3f' % gs.best_score_)
    # predicción en test con los mejores parámetros
    y_pred = gs.predict(X_test)
    # mejor accuracy en test con los mejores parámetros
    print('Mejor accuracy en test con los mejores parámetros: %.3f ' % accuracy_score(y_test, y_pred))
    # mejor modelo (por accuracy)
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClasificador con el mejor accuracy: %s' % grid_dict[best_clf])
