#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------------------------------------------------
# I. IMPORTACIÓN DE LIBRERÍAS Y MÓDULOS

# Para manipulación y análisis de datos.
import pandas as pd 
import numpy as np

# Para visualización de datos mediante la generación de gráficos.
import matplotlib.pyplot as plt 
import seaborn as sns 

# Para implementar modelos de machine learning.
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve
from sklearn import set_config
set_config(display = 'diagram')

# Para recodificación de atributos
from category_encoders import TargetEncoder, OrdinalEncoder, OneHotEncoder

from datetime import datetime

# Para visualización
import graphviz

# Para preprocesar datos perdidos.
import missingno as msngo

# Para serialización
import pickle

# Para evitar mensajes de deprecación
import warnings 
warnings.filterwarnings("ignore")

import pprint
import csv
import re

#-----------------------------------------------------------------------------------------------------------------------
# I. DEFINICIÓN DE FUNCIONES

# 1.Función para recodificar y/o imputar en forma directa, a partir de una lista dada.
def recod_where(dataframe, variable, lista, termino):
    """
    recod_where - Permite recodificar y/o imputar los datos de una variable con un término dado, 
    en función de una lista, por medio de np.where.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por analizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - variable: Parámetro obligatorio. Variable en la cual se desean recodificar los términos contenidos en la lista entregada.
        - lista: Parámetro obligatorio. Lista que contiene los nombres de las variables (como string) 
                 que se desea recorrer para aplicar la recodificación.
        - termino: Parámetro obligatorio. Término a utillizar para realizar la imputación o recodificación necesaria. 

    @salida:
        - Columna reescrita recodificada.
    """
    for i in lista:
        dataframe[variable] = np.where(dataframe[variable] == i, termino, dataframe[variable])


# 2.Función para conteo de la cantidad de observaciones.
def conteo(dataframe, variable):
    """
    conteo - Permite contar la cantidad de ocurrencias para los diferentes datos existentes en una variable determinada.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por analizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - variable: Parámetro obligatorio. Variable sobre la cual se desea realizar el conteo de observaciones.

    @salida:
        - Cantidad de datos en una variable, para sus distintas observaciones.
    """
    return dataframe[variable].value_counts()


# 3. Función para reemplazar espacios en blanco por datos np.NaN
def replace_nan(dataframe, list_nan):
    """
    replace_nan - Permite reemplazar los valores dados en una lista por datos de tipo np.NaN

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por utilizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - list_nan: Parámetro obligatorio. Lista que contiene los datos que se desean reemplazar en el dataframe.

    @salida:
        - Dataframe con datos np.NaN, imputados de acuerdo a la lista entregada.
    """

    columns_tmp = dataframe.columns
    
    for i in columns_tmp:
        dataframe[i] = dataframe[i].replace(list_nan, np.nan)

# 4. Función para crear subsets a partir del dataframe "LINIO":
def crear_subset_linio(dataframe_original, lista_variables):
    """
    crear_subset_linio - Permite crear subsets del dataframe 'linio', a partir de la selección 
    de ciertas columnas del dataframe original.

    @parámetros:
        - dataframe_original: Parámetro obligatorio. Set de datos que contiene todos los atributos y/o variables originales,
          correspondiente al tipo pandas.core.frame.DataFrame.
        - lista_variable: Parámetro obligatorio. Lista que contiene las variables que se desea incorporar en el dataframe,
          en forma adicional a aquellas predefinidas en la función.

    @salida:
        - Subset de datos.
    """
    lista_variables_fijas = ['retail','categoria','subcategoria','codigo', 'producto', 'precio_original',
                             'url_foto','marca','marca_2', 'descripcion','por_definir', 'descripcion_2',
                             'valor_internet', 'valor_oferta','usuario']
    
    tmp = lista_variables_fijas + lista_variables
    
    nuevo_dataframe = dataframe_original.loc[:, tmp]
    
    return nuevo_dataframe

# 5. Función para crear subsets a partir del dataframe "RIPLEY":
def crear_subset_ripley(dataframe_original, lista_variables):
    """
    crear_subset_ripley - Permite crear subsets del dataframe 'ripley', a partir de la selección 
    de ciertas columnas del dataframe original.

    @parámetros:
        - dataframe_original: Parámetro obligatorio. Set de datos que contiene todos los atributos y/o variables originales,
          correspondiente al tipo pandas.core.frame.DataFrame.
        - lista_variable: Parámetro obligatorio. Lista que contiene las variables que se desea incorporar en el dataframe,
          en forma adicional a aquellas predefinidas en la función.

    @salida:
        - Subset de datos.
    """
    lista_variables_fijas = ['retail','categoria','producto','precio_original','precio_internet','precio_oferta','categoria_2']
    
    tmp = lista_variables_fijas + lista_variables
    
    nuevo_dataframe = dataframe_original.loc[:, tmp]
    
    return nuevo_dataframe

# 5. Función para crear subsets a partir del dataframe "PARIS":
def crear_subset_paris(dataframe_original, lista_variables):
    """
    crear_subset_paris - Permite crear subsets del dataframe 'paris', a partir de la selección 
    de ciertas columnas del dataframe original.

    @parámetros:
        - dataframe_original: Parámetro obligatorio. Set de datos que contiene todos los atributos y/o variables originales,
          correspondiente al tipo pandas.core.frame.DataFrame.
        - lista_variable: Parámetro obligatorio. Lista que contiene las variables que se desea incorporar en el dataframe,
          en forma adicional a aquellas predefinidas en la función.

    @salida:
        - Subset de datos.
    """
    lista_variables_fijas = ['retail','categoria','producto','precio_original','precio_internet','precio_oferta']
    
    tmp = lista_variables_fijas + lista_variables
    
    nuevo_dataframe = dataframe_original.loc[:, tmp]
    
    return nuevo_dataframe

# 6. Función para homologar nombres de los atributos de los subsets creados - LINIO
def homologar_nombres_linio(lista_dataframes):
    """
    homologar_nombres_linio - Permite homologar los nombres de los atributos de los subsets creados a partir del dataframe original.

    @parámetros:
        - lista_dataframes: Parámetro obligatorio. Lista que contiene los subsets cuyos nombres de columnas/atributos 
          deben ser homologados, para su posterior consolidación.

    @salida:
        - Subsets de datos definidos en lista con nombres de columnas homologados.
    """
    for i in lista_dataframes:
        i.columns = ['retail','categoria','subcategoria','codigo', 'producto', 'precio_original', 'url_foto','marca',
                  'marca_2', 'descripcion', 'por_definir', 'descripcion_2','valor_internet', 'valor_oferta',
                  'autor','rating','comentario']

# 7. Función para homologar nombres de los atributos de los subsets creados - RIPLEY
def homologar_nombres_ripley(lista_dataframes):
    """
    homologar_nombres_ripley - Permite homologar los nombres de los atributos de los subsets creados a partir del dataframe original.

    @parámetros:
        - lista_dataframes: Parámetro obligatorio. Lista que contiene los subsets cuyos nombres de columnas/atributos 
          deben ser homologados, para su posterior consolidación.

    @salida:
        - Subsets de datos definidos en lista con nombres de columnas homologados.
    """
    for i in lista_dataframes:
        i.columns = ['retail','categoria','producto','precio_original','precio_internet','precio_oferta','categoria_2','fecha',
                     'usuario','rating','comentario']

# 8. Función para homologar nombres de los atributos de los subsets creados - PARIS
def homologar_nombres_paris(lista_dataframes):
    """
    homologar_nombres_paris - Permite homologar los nombres de los atributos de los subsets creados a partir del dataframe original.

    @parámetros:
        - lista_dataframes: Parámetro obligatorio. Lista que contiene los subsets cuyos nombres de columnas/atributos 
          deben ser homologados, para su posterior consolidación.

    @salida:
        - Subsets de datos definidos en lista con nombres de columnas homologados.
    """
    for i in lista_dataframes:
        i.columns = ['retail','categoria','producto','precio_original','precio_internet','precio_oferta',
                     'fecha','usuario','rating','comentario']

# 8. Función para renombrar datos de las variables
def renombrar_datos(dataframe,variable,termino_a_reemplazar,nuevo_termino):
    """
    renombrar_datos - Permite renombrar los datos de una variable con un término dado, por medio de np.where.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por analizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - variable: Parámetro obligatorio. Variable en la cual se desean renombrar términos.
        - termino_a_reemplazar: Parámetro obligatorio. Término que desea ser reemplazado.
        - nuevo_termino: Parámetro obligatorio. Término a utillizar para realizar el reemplazo requerido. 

    @salida:
        - Columna con datos renombrados.
    """    
    dataframe[variable] = np.where(dataframe[variable] == termino_a_reemplazar, nuevo_termino, dataframe[variable])

# 9. Función para consolidar o concatenar dataframes
def concatenar_dataframes(lista_dataframes):
    """
    concatenar_dataframes - Permite consolidar o concatenar una serie de subsets, definidos previamente en una lista.

    @parámetros:
        - lista_dataframes: Parámetro obligatorio. Lista que contiene los subsets a ser consolidados.

    @salida:
        - Nuevo dataframe, confeccionado a partir de los subsets declarados en lista.
    """
    df_concat = pd.concat(lista_dataframes, axis=0)
    return df_concat

# 10. Función para reemplazar comillas dobles de columnas categóricas:
def quitaComillaDoble(dataframe,columna):
    """
    quita_comillas_dobles - Permite eliminar el caracter correspondiente a comillas dobles, de los datos de una columna específica.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por utilizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - columna: Parámetro obligatorio. Variable en cuyas observaciones se requiere quitar las comillas.

    @salida:
        - Dataframe con observaciones sin comillas.
    """
    for i in dataframe[columna]:

        dataframe[columna] = dataframe[columna].str.replace('"', '')

        return dataframe[columna]

# 11. # Función para preprocesar los atributos "valor_internet"/"valor_oferta"
def proc_str_num(dataframe, atrib, caract, num_conv = False):
    """
    proc_str_num - Permite procesar atributos para remover caracteres específicos.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por utilizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - atrib: Parámetro obligatorio. Variable a la cual se aplicará el preprocesamiento.
        - caract: Parámetro obligatorio. Tipo de caracter que requiere ser reemplazado y procesado.
        - num_conv: Parámetro obligatorio. Indica a la función si se requiere preprocesamiento adicional 
                    para la conversión del tipo de dato.

    @salida:
        - Dataframe con datos procesados según requerimientos.
    """
    dataframe[atrib] = dataframe[atrib].str.replace('"','')
    dataframe[atrib] = dataframe[atrib].replace('',np.nan)
    dataframe[atrib] = dataframe[atrib].replace(np.nan,'0.0')
    
    for i in range(len(dataframe[atrib])):
        dataframe[atrib][i]= re.sub('['+caract+']','', dataframe[atrib][i])
    
    if num_conv:
        dataframe[atrib] = dataframe[atrib].astype(int)
        dataframe[atrib] = dataframe[atrib].replace('0',np.nan)
    return dataframe[atrib]

# 12. Función para la inspección visual de variables categóricas
def inspeccion(dataframe,variable):
    """
    inspeccion - Permite inspeccionar variables categóricas mediante gráficos que dan cuenta de su frecuencia.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por utilizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - variable: Parámetro obligatorio. Variable categórica a ser explorada en forma visual.

    @salida:
        - Gráfico de frecuencias para variables categóricas.
    """
    var = dataframe[variable].unique()

    frecuencias = dataframe[variable].value_counts()
    plt.bar(var, frecuencias,edgecolor='black', align="center")
    plt.title("Frecuencia del {} de los productos".format(variable))
    plt.xticks(var, var, rotation= "vertical")
    plt.xlabel(variable)
    plt.ylabel("frecuencias")
    plt.show()
   
# 13. Función para la inspección visual de variables continuas
def plot_hist(data, variable):
    """
    plot_hist - Permite inspeccionar variables continuas mediante gráficos.

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por utilizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - variable: Parámetro obligatorio. Variable a ser explorada en forma visual.

    @salida:
        - Gráfico de histograma para variables continuas.
    """

    plt.title("Histograma de la variable " +variable)
    columna= data[variable].dropna()
    
    plt.hist(columna, color= "purple")
    plt.axvline(columna.mean(), linewidth=3, color="black",linestyle="--", label="Media")
    plt.axvline(columna.median(), linewidth=3, color="lightcoral",linestyle="--", label="Mediana")
    
    plt.legend()