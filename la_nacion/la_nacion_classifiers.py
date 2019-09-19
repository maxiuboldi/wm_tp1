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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

SEED = 15
K_FOLD = 5
JOBS = 4
N_ITER = 1000

print('Leyendo dataset')
# Leer el dataset
dataset = pd.read_pickle(r'la_nacion\datasets\la_nacion_dataset.pkl')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# clases a números
le = LabelEncoder()
y = le.fit_transform(y)
# números y categorias
num_clase = le.classes_

print('\nSeparando en traint y test')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

print('\nGenerando Pipeline')
pipe = Pipeline([('scl', SelectKBest(score_func=chi2)),
                 ('clf', SVC(random_state=SEED))])

# esta forma de definir el espacio de búsqueda con una lista en RandomizedSearchCV está disponible a partir de la versión Version 0.22.dev0.
# pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
search_space = [{'clf': [LogisticRegression(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200, 250],
                 'clf__penalty': ['l1', 'l2'],
                 'clf__C': np.logspace(-4, 4, 50),
                 'clf__solver': ['liblinear'],
                 'clf__max_iter': range(100, 500, 50),
                 'clf__multi_class': ['auto']},
                {'clf': [RandomForestClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200, 250],
                 'clf__criterion': ['gini', 'entropy'],
                 'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 'clf__max_depth': range(3, 30, 2),
                 'clf__min_samples_split': [2, 5, 10],
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
                {'clf': [SVC(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200, 250],
                 'clf__kernel': ['linear', 'rbf', 'poly'],
                 'clf__degree': [0, 1, 2, 3, 4, 5, 6],
                 'clf__gamma': ['scale', 'auto'],
                 'clf__decision_function_shape': ['ovo', 'ovr'],
                 'clf__C': np.linspace(0.001, 200, 50)}]

gs = RandomizedSearchCV(estimator=pipe,
                        param_distributions=search_space,
                        scoring='accuracy',
                        cv=K_FOLD,
                        n_jobs=JOBS,
                        verbose=1,
                        return_train_score=True,
                        n_iter=N_ITER)

print('\nAjustando Pipeline en la búsqueda con CV\n')
gs.fit(X_train, y_train)

# resultados del CV
cv_result = pd.DataFrame(gs.cv_results_)

# features finales y pesos
names = X_train.columns.values[gs.best_estimator_.named_steps["scl"].get_support()]
scores = gs.best_estimator_.named_steps["scl"].scores_[gs.best_estimator_.named_steps["scl"].get_support()]
names_scores = list(zip(names, scores))
features_sel = pd.DataFrame(names_scores, columns=['Feature', 'Scores'])

# predicción del resultado
y_pred = gs.predict(X_test)

# métricas para exportar
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=num_clase, index=num_clase)
class_rep = pd.DataFrame.from_dict(classification_report(y_test, y_pred, target_names=num_clase, output_dict=True))

print('\nMatriz de Confusión')
print(conf_mat)
print('\nReporte de Clasificación')
print(classification_report(y_test, y_pred, target_names=num_clase))
print('\nAccuracy_score: {}'.format(accuracy_score(y_test, y_pred)))

print('\nExportando resultados')
with pd.ExcelWriter(r'la_nacion\resultados\resultados_la_nacion.xlsx') as writer:
    cv_result.to_excel(writer, sheet_name='CV_Resultado', index=False)
    features_sel.to_excel(writer, sheet_name='Features_Seleccionadas', index=False)
    conf_mat.to_excel(writer, sheet_name='Matriz_Confusion', index=False)
    class_rep.to_excel(writer, sheet_name='Reporte_Clasificacion', index=False)

joblib.dump(gs.best_estimator_, r'la_nacion\resultados\modelo_la_nacion.pkl')

print('\nListo!')

# para probar
# model = joblib.load(r'la_nacion\resultados\modelo_la_nacion.pkl')
# y_pred2 = model.predict(X_test)
#
# print('\nMatriz de Confusión')
# print(pd.DataFrame(confusion_matrix(y_test, y_pred2), columns=num_clase, index=num_clase))
# print('\nReporte de Clasificación')
# print(classification_report(y_test, y_pred2, target_names=num_clase))
# print('\nAccuracy_score: {}'.format(accuracy_score(y_test, y_pred2)))
