import pandas as pd
import numpy as np
import caracteristicasDerivadas
from caracteristicasDerivadas import H_entropy, proporcionVocalesConsonantes, posicionPrimerDigito
import  matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
from pandas_profiling import ProfileReport

import sklearn
from sklearn import metrics, model_selection, tree

#Carga de datos
df = pd.read_csv('../data/dga_data_small.csv')

#Preprocesamiento

#Eliminacion de caracteristicas irrelevante o repetidas
df.drop(['host','subclass'], axis=1, inplace=True)

#Codificacion de variable objetivo
df['isDGA'] = df['isDGA'].replace(to_replace='dga', value=1)
df['isDGA'] = df['isDGA'].replace(to_replace='legit', value=0)


#Derivar caracteristicas
# 1. Longitud
# 2. Cantidad de digitos
# 3. Calculo de entropia (Shannon), es decir la cantidad de "informacion" que se puede obtener de una variable
# 4. Posicion del primer digito en la cadena
# 5. Proporcion de vocales - consonantes

df['longitud'] = df['domain'].str.len()
df['digitos'] = df['domain'].str.count('[0-9]')
df['entropia'] = df['domain'].apply(H_entropy)
df['proporcionVocalesConsonantes'] = df['domain'].apply(proporcionVocalesConsonantes)
df['posicionPrimerDigito'] = df['domain'].apply(posicionPrimerDigito)

df.drop(['domain'], axis=1, inplace=True)



print('Final features:', df.columns)
print(df.head())

df_final = df

#Visualizacion de informacion
#nombres_variables = ['longitud', 'digitos', 'entropia', 'proporcionVocalesConsonantes', 'posicionPrimerDigito']
#sns.pairplot(df_final, hue='isDGA', vars=nombres_variables)
#plt.show()

#profile = ProfileReport(df_final, title='Reporte DGA final')
#profile.to_file('Reporte datos DGA presentacion.html')


target = df_final['isDGA']
feature_matrix = df_final.drop(['isDGA'], axis=1)

print('Final features:', feature_matrix.columns)
feature_matrix.head()

feature_matrix_train, feature_matrix_test, target_train, target_test = model_selection.train_test_split(feature_matrix, target, test_size=0.25, random_state=31)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature_matrix_train, target_train)

import joblib
joblib.dump(clf, 'clf.joblib')

print(feature_matrix_train.count())

print(feature_matrix_test.count())

#Metricas

target_pred = clf.predict(feature_matrix_test)

print(metrics.accuracy_score(target_test, target_pred))
print('Matriz de confusion /n',metrics.confusion_matrix(target_test, target_pred))
print(metrics.classification_report(target_test, target_pred, target_names=['legit', 'dga']))






