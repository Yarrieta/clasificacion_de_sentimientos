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
from wordcloud import WordCloud

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

# Procesamiento de Texto
from spellchecker import SpellChecker
spell = SpellChecker(language='es')
from sklearn.feature_extraction.text import CountVectorizer
import spacy
nlp = spacy.load('es_core_news_sm')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('spanish')
from string import punctuation
import re

import pprint
import csv

#-----------------------------------------------------------------------------------------------------------------------
# I. DEFINICIÓN DE NOMBRES DE ATRIBUTOS

# 1. Atributos Linio
atrib_linio = ['categoria','codigo','producto','precio_original','url_foto','marca', 'marca_2','descripcion','por_definir',
           'descripcion_2','comentario_1','precio_internet','precio_oferta','usuario','comentario_2','comentario_3',
           "comentario_4",'comentario_5','comentario_6','comentario_7','comentario_8','comentario_9','comentario_10',
           'rating_1','rating_2','rating_3','rating_4','rating_5','rating_6','rating_7','rating_8','rating_9',
           'rating_10','subcategoria']

# 2. Atributos Ripley
atrib_ripley = ['categoria','codigo','producto','precio_original','url_imagen','en_blanco','marca',
           'descripcion_1','en_blanco2','descripcion_2','comentario_1',
           'precio_internet','precio_oferta','usuario_1','comentario_2',
           'comentario_3','comentario_4','comentario_5','comentario_6','comentario_7',
           'comentario_8','comentario_9','comentario_10','rating_1','rating_2','rating_3','rating_4',
           'rating_5','rating_6','rating_7','rating_8','rating_9','rating_10',
           'categoria_2','url_producto','fecha_1','fecha_2','fecha_3','fecha_4','fecha_5','fecha_6','fecha_7',
           'fecha_8','fecha_9','fecha_10','fecha_11','fecha_12','fecha_13','fecha_14','fecha_15',
           'usuario_2','usuario_3','usuario_4','usuario_5','usuario_6','usuario_7','usuario_8',
           'usuario_9','usuario_10','usuario_11','usuario_12','usuario_13','usuario_14',
           'usuario_15','rating_11','rating_12','rating_13','rating_14','rating_15','comentario_11','comentario_12',
           'comentario_13', 'comentario_14', 'comentario_15']

# 3. Atributos Paris
atrib_paris = ['retail', 'codigo', 'producto', 'precio_original', 'url_imagen', 'vendedor', 'url_producto',
          'modelo', 'descripcion', 'comentario_1', 'precio_internet', 'precio_oferta', 'usuario_1', 
          'comentario_2', 'comentario_3', 'comentario_4', 'comentario_5', 'comentario_6', 'comentario_7', 
          'comentario_8', 'comentario_blanco_1', 'comentario_blanco_2', 'rating_1', 'rating_2', 'rating_3',
          'rating_4', 'rating_5', 'rating_6','rating_7', 'rating_8', 'rating_blanco_1', 'rating_blanco_2', 
          'en_blanco', 'en_blanco_2', 'fecha_comentario_1','fecha_comentario_2', 'fecha_comentario_3', 
          'fecha_comentario_4', 'fecha_comentario_5', 'fecha_comentario_6', 'fecha_comentario_7', 
          'fecha_comentario_8', 'en_blanco_3','en_blanco_4', 'en_blanco_5', 'en_blanco_6', 'en_blanco_7', 
          'en_blanco_8', 'en_blanco_9', 'usuario_2', 'usuario_3', 'usuario_4', 'usuario_5', 'usuario_6', 
          'usuario_7','usuario_8', 'en_blanco_10', 'en_blanco_22', 'en_blanco_11','en_blanco_12', 
          'en_blanco_13', 'en_blanco_14', 'en_blanco_15', 'rating_blanco_11', 'rating_blanco_12', 
          'rating_blanco_13', 'rating_blanco_14', 'rating_blanco_15', 'comentario_blanco_11', 
          'comentario_blanco_12','comentario_blanco_13', 'comentario_blanco_14', 'comentario_blanco_15', 
          'precio_2', 'categoria']

#-----------------------------------------------------------------------------------------------------------------------
# I. DEFINICIÓN DE CATEGORIAS

# 1. Categorias de Linio
games = ["videojuegos_total.txt\"","videojuegos.txt\"", "videojuegos3\"","videojuegos2\"", 'videojuegos"']
tec = ['celulares"', 'tablets"', "tv_audio_video_2\"","tv_audio_video_total_total.txt\"", "tv_audio_video.txt\""]
elec = ["linea_blanca.txt\"","linea_blanca2\"", 'linea_blanca"']


# 2. Categorias Ripley
belleza = ['perfumeria1.gsd"','perfumeria2.gsd"', 'perfumeria3.gsd"','perfumeria4.gsd"','perfumeria5.gsd"',
           'perfumeria6.gsd"','maquillaje_12.gsd"','maquillaje_14.gsd"','maquillaje_13.gsd"','maquillaje_22.gsd"',
           'maquillaje_21.gsd"','maquillaje_11.gsd"','maquillaje_23.gsd"','maquillaje_15.gsd"','belleza_cuerpo"']

tecnologia = ['tecnologia_17.gsd"','tecnologia_110.gsd"','tecnologia_111.gsd"','tecnologia_13.gsd"',
              'tecnologia_19.gsd"','tecnologia_14.gsd"','tecnologia_18.gsd"','tecnologia_11.gsd"',
              'tecnologia_12.gsd"','tecnologia_16.gsd"', 'tecnologia_15.gsd"','tecnologia_112.gsd"',
              'fotografia1.gsd"','fotografia2.gsd"']

computacion = ['computacion1.gsd"',
              'computacion2.gsd"','computacion3.gsd"']

videojuegos = ['tecno_gamer1.gsd"', 'tecno_gamer2.gsd"', 'tecno_gamer3.gsd"']

jugueteria = ['jugueteria2.gsd"','jugueteria5.gsd"','jugueteria1.gsd"','jugueteria4.gsd"','jugueteria3.gsd"',
              'jugueteria6.gsd"']

fitness = ['fitnes8.gsd"','fitnes4.gsd"','fitnes6.gsd"','fitnes3.gsd"','fitnes1.gsd"','fitnes2.gsd"',
           'fitnes5.gsd"','fitnes9.gsd"','fitnes2.gsd"','fitnes7.gsd"','fitness2"']

supermercado = ['supermercado2.gsd"','supermercado1.gsd"','supermercado3.gsd"','supermercado4.gsd"']

accesorios = ['disk2.gsd"','disk1.gsd"']

joyas = ['joyas_513.gsd"','joyas_39.gsd"','joyas_12.gsd"','joyas_74.gsd"','joyas_73.gsd"','joyas_87.gsd"',
         'joyas_1.gsd"','joyas_25.gsd"','joyas_88.gsd"','joyas_912.gsd"','joyas_26.gsd"','joyas_310.gsd"',
         'joyas_311.gsd"','joyas_515.gsd"','joyas_617.gsd"','joyas_619.gsd"','joyas_618.gsd"','joyas_516.gsd"',
         'joyas_620.gsd"','joyas_621.gsd"']

moda = ['chaquetas4.gsd"','chaquetas6.gsd"','chaquetas13.gsd"','chaquetas1.gsd"','chaquetas3.gsd"',
        'chaquetas15.gsd"','chaquetas5.gsd"','chaquetas2.gsd"','chaquetas7.gsd"','chaquetas11.gsd"',
        'chaquetas14.gsd"','chaquetas12.gsd"','chaquetas10.gsd"','chaquetas8.gsd"','chaquetas16.gsd"',
        'jeans_28.gsd"','jeans_13.gsd"','jeans_11.gsd"','jeans_12.gsd"','jeans_27.gsd"','jeans_14.gsd"',
        'jeans_25.gsd"','jeans_26.gsd"','jeans_29.gsd"']

hogar = ['hogar_22.gsd"','hogar_12.gsd"','hogar_21.gsd"','hogar_11.gsd"','hogar_3.gsd"','hogar_23.gsd"',
         'hogar_13.gsd"', 'hogar_3"', 'dormitorio"','dormitorio_21.txt"','dormitorio_22.txt"','dormitorio_24.txt"',
         'dormitorio_23.txt"']

electro = ['electro"','electro1.gsd"','electro2.gsd"']


# 3. Categorias Paris
elec_p = ['"linea-blanca/electrodomesticos/','"linea-blanca/refrigeracion/','"linea-blanca/cocina/',
                '"linea-blanca/lavado-secado/','"linea-blanca/marcas/']
mujer_p = ['"mujer/moda/', '"mujer/colecciones/','"mujer/ropa-interior/','"mujer/marcas/']

tecnologia_p = ['"tecnologia/celulares/','"tecnologia/marcas/', '"tecnologia/smart-home/','"tecnologia/fotografia/',
              '"electro/audio/', '"electro/television/','"electro/audio-hifi/','"electro/instrumentos-musicales/',
              '"electro/drones/','"electro/marcas/','"electro/accesorios-auto/', '"']

computacion_p = ['"tecnologia/computadores/', '"tecnologia/accesorios-computacion/']

gamer_p = ['"tecnologia/consolas-videojuegos/', '"tecnologia/gamers/']

jugueteria_p = ['"jugueteria/tecnologia-ninos/']

hogar_p = ['"dormitorio/ropa-cama/']

moda_p = ['"hombre/moda/'] 

#-----------------------------------------------------------------------------------------------------------------------
# I. DEFINICIÓN DE FUNCIONES PREPROCESAMIENTO

# 1.1 Funcion de creacion de subset de LINIO

def preproc_lineo(dataframe_original):
    """
    crear_subset_linio - Permite crear subsets del dataframe 'linio', a partir de la selección 
    de ciertas columnas del dataframe original, las cuales han sido estblecidas dentro de la presente funcion.

    @parámetros:
        - dataframe_original: Parámetro obligatorio. Set de datos que contiene todos los atributos y/o variables originales,
          correspondiente al tipo pandas.core.frame.DataFrame.

    @salida:
        - Subset de datos.
    """
    # Creacion de atributo faltante
    dataframe_original['retail'] = 'linio'
    
    # Homologaciones de atributo categoria
    recod_where(dataframe_original,'categoria',games,"videojuegos")
    recod_where(dataframe_original,'categoria',tec,"tecnologia")
    recod_where(dataframe_original,'categoria',elec,"electro")
    
    # Depuracion de atributo categoria
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('belleza"', 'belleza')
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('libros"', 'libros')
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('deportes"', 'deportes')
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('moda"', 'moda')
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('computacion"', 'computacion')
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('hogar"', 'hogar')
    dataframe_original['categoria'] = dataframe_original['categoria'].replace('ferreteria"', 'ferreteria')
    
    # Atributos definidos como fijos
    fix_feat = ['retail','categoria','codigo', 'producto', 'precio_original', 'precio_internet', 'precio_oferta','usuario']
    
    # Atributos variables a ser agrupados por comentario
    stacks_feat = [['comentario_1', 'rating_1'], ['comentario_2', 'rating_2'], ['comentario_3', 'rating_3'],
                   ['comentario_4', 'rating_4'], ['comentario_5', 'rating_5'], ['comentario_6', 'rating_6'],
                   ['comentario_7', 'rating_7'], ['comentario_8', 'rating_8'], ['comentario_9', 'rating_9'],
                   ['comentario_10', 'rating_10']]
    
    # Generacion de un dataframe por cada comentario. Posteriormente seran concatenados para formar un unico dataframe
    tmp = [fix_feat+stack for stack in stacks_feat]
    
    list_df = []
    tabla = dict()
    
    for indice, lista in enumerate(tmp):
        tabla['tabla_'+str(indice+1)] = dataframe_original.loc[:, lista]
        list_df.append(tabla['tabla_'+str(indice+1)])

    
    for tabla in list_df:
        tabla.columns = ['retail','categoria','codigo', 'producto', 'precio_original', 'precio_internet', 'precio_oferta','usuario',
                          'comentario', 'rating']
    

    # Concatenacion de dataframes generados
    df_concat = pd.concat(list_df, axis=0)
    
    # Reemplazo de valores vacios por nulos
    nan = ['""','" "','"        "','"         "','"            "','"                      "']
    df_concat = df_concat.replace(nan, np.nan)
    
    # Eliminacion de comillas dobles
    df_concat = df_concat.applymap(lambda x: x.replace('"', '') if type(x) is str else x)
    
    # Eliminacion de puntos
    df_concat = df_concat.applymap(lambda x: x.replace('.', '') if type(x) is str else x)
     
    # Modificacion de precios a enteros
    df_concat['precio_original'] = df_concat['precio_original'].replace(np.nan, '-99')
    df_concat['precio_original'] = df_concat['precio_original'].astype(float)
    df_concat['precio_original'] = df_concat['precio_original'].replace(-99, np.nan)
    
    df_concat['precio_internet'] = df_concat['precio_internet'].replace(np.nan, '-99')
    df_concat['precio_internet'] = df_concat['precio_internet'].astype(float)
    df_concat['precio_internet'] = df_concat['precio_internet'].replace(-99, np.nan)    

    df_concat['precio_oferta'] = df_concat['precio_oferta'].replace(np.nan, '-99')
    df_concat['precio_oferta'] = df_concat['precio_oferta'].astype(float)
    df_concat['precio_oferta'] = df_concat['precio_oferta'].replace(-99, np.nan)
    
    # Modificacion de Rating a Float (provisional por presencia de datos nulos)
    df_concat['rating'] = df_concat['rating'].replace(np.nan, '-99')
    df_concat['rating'] = df_concat['rating'].astype(float)
    df_concat['rating'] = df_concat['rating'].replace(-99, np.nan)   

    
    # Modificacion de fechas corruptas
    # En el caso de linio, el atributo autor contiene al usuario y la fecha.
    df_concat['fecha'] = df_concat['usuario']
    df_concat['fecha'] = df_concat['fecha'].replace(np.nan, 'S_I')
    df_concat['fecha'] = df_concat['fecha'].map(lambda x: x.replace(' ', ''))
    df_concat['fecha'] = [re.sub(r"[a-zA-ZÌ@£_]", '', i) for i in df_concat['fecha']]
    df_concat['fecha'] = df_concat['fecha'].str.replace('0000000000', '')
    df_concat['fecha'] = df_concat['fecha'].str.replace('245', '')
    df_concat['fecha'] = df_concat['fecha'].str.replace('906/11/20', '06/11/20')
    df_concat['fecha'] = df_concat['fecha'].str.replace('7727/12/20', '27/12/20')
    df_concat['fecha'] = df_concat['fecha'].str.replace('36306/09/20', '06/09/20')
    df_concat['fecha'] = df_concat['fecha'].str.replace('216/12/20', '06/09/20')
    df_concat['fecha'] = df_concat['fecha'].str.replace('009/01/21', '09/01/21')
    df_concat['fecha'] = df_concat['fecha'].str.replace('009/01/21', '09/01/21')
    df_concat['fecha'] = df_concat['fecha'].str.replace('140204/01/21', '04/01/21')
    
    # Modificacion a formato datetime para extraccion de meses y años
    df_concat['fecha'] = pd.to_datetime(df_concat['fecha'])
    df_concat['fecha'] = [d.date() for d in df_concat['fecha']]
    df_concat['año'] = pd.DatetimeIndex(df_concat['fecha']).year
    df_concat['mes'] = pd.DatetimeIndex(df_concat['fecha']).month
    
    # Eliminacion de columnas no requeridas
    df_concat = df_concat.drop(columns = ['usuario', 'fecha'])

    # Eliminación de registros duplicados en el dataset
    df_concat = df_concat.drop_duplicates() 
    df_concat.reset_index(drop=True, inplace=True)
    
    return df_concat
    
# return tabla['tabla_1'], tabla['tabla_2'], tabla['tabla_3'], tabla['tabla_4'], tabla['tabla_5'], tabla['tabla_6'], tabla['tabla_7'], tabla['tabla_8'], tabla['tabla_9'], tabla['tabla_10']

# 1.2 Funcion de creacion de subset de RIPLEY

def preproc_ripley(dataframe_original):
    """
    crear_subset_ripley - Permite crear subsets del dataframe 'ripley', a partir de la selección 
    de ciertas columnas del dataframe original, las cuales han sido estblecidas dentro de la presente funcion.

    @parámetros:
        - dataframe_original: Parámetro obligatorio. Set de datos que contiene todos los atributos y/o variables originales,
          correspondiente al tipo pandas.core.frame.DataFrame.

    @salida:
        - Subset de datos.
    """
    # Creacion de atributo faltante
    dataframe_original['retail'] = 'ripley'
    
    
    # Homologaciones de categorias
    recod_where(dataframe_original,'categoria',belleza,"belleza")
    recod_where(dataframe_original,'categoria',tecnologia,"tecnologia")
    recod_where(dataframe_original,'categoria',videojuegos,"videojuegos")
    recod_where(dataframe_original,'categoria',jugueteria,"jugueteria")
    recod_where(dataframe_original,'categoria',fitness,"fitness")
    recod_where(dataframe_original,'categoria',supermercado,"supermercado")
    recod_where(dataframe_original,'categoria',accesorios,"accesorios")
    recod_where(dataframe_original,'categoria',joyas,"joyas")
    recod_where(dataframe_original,'categoria',moda,"moda")
    recod_where(dataframe_original,'categoria',hogar,"hogar")
    recod_where(dataframe_original,'categoria',electro,"electro")
    recod_where(dataframe_original,'categoria',computacion,"computacion")
    
    fix_feat = ['retail','categoria','codigo', 'producto','precio_original','precio_internet','precio_oferta']
    
    stacks_feat = [['comentario_1', 'rating_1', 'fecha_1'], ['comentario_2', 'rating_2', 'fecha_2'],
                   ['comentario_3', 'rating_3', 'fecha_3'], ['comentario_4', 'rating_4', 'fecha_4'],
                   ['comentario_5', 'rating_5', 'fecha_5'], ['comentario_6', 'rating_6', 'fecha_6'],
                   ['comentario_7', 'rating_7', 'fecha_7'], ['comentario_8', 'rating_8', 'fecha_8'],
                   ['comentario_9', 'rating_9', 'fecha_9'], ['comentario_10', 'rating_10', 'fecha_10'],
                   ['comentario_11', 'rating_11', 'fecha_11'], ['comentario_12', 'rating_12', 'fecha_12'],
                   ['comentario_13', 'rating_13', 'fecha_13'], ['comentario_14', 'rating_14', 'fecha_14'],
                   ['comentario_15', 'rating_15', 'fecha_15']]
    
    # Generacion de un dataframe por cada comentario. Posteriormente seran concatenados para formar un unico dataframe
    tmp = [fix_feat+stack for stack in stacks_feat]
    
    list_df = []
    tabla = dict()
    
    for indice, lista in enumerate(tmp):
        tabla['tabla_'+str(indice+1)] = dataframe_original.loc[:, lista]
        list_df.append(tabla['tabla_'+str(indice+1)])
    
    for tabla in list_df:
        tabla.columns = ['retail','categoria','codigo', 'producto', 'precio_original', 'precio_internet', 'precio_oferta','comentario',
                          'rating', 'fecha']
    

    # Concatenacion de dataframes generados
    df_concat = pd.concat(list_df, axis=0)
    
    # Reemplazo de valores vacios por nulos
    nan = ['""','" "','"        "','"         "','"            "','"                      "']
    df_concat = df_concat.replace(nan, np.nan)
    
    # Eliminacion de comillas dobles
    df_concat = df_concat.applymap(lambda x: x.replace('"', '') if type(x) is str else x)
    
    # Eliminacion de puntos
    df_concat = df_concat.applymap(lambda x: x.replace('.', '') if type(x) is str else x)
    
    # Eliminacion de registros No disponible en precio_internet
    df_concat['precio_internet'] = df_concat['precio_internet'].replace('No disponible', np.nan)
     
    # Modificacion de precios a enteros
    df_concat['precio_original'] = df_concat['precio_original'].replace(np.nan, '-99')
    df_concat['precio_original'] = df_concat['precio_original'].astype(float)
    df_concat['precio_original'] = df_concat['precio_original'].replace(-99, np.nan)
    
    df_concat['precio_internet'] = df_concat['precio_internet'].replace(np.nan, '-99')
    df_concat['precio_internet'] = df_concat['precio_internet'].astype(float)
    df_concat['precio_internet'] = df_concat['precio_internet'].replace(-99, np.nan)    
    
    df_concat['precio_oferta'] = df_concat['precio_oferta'].replace(np.nan, '-99')
    df_concat['precio_oferta'] = df_concat['precio_oferta'].astype(float)
    df_concat['precio_oferta'] = df_concat['precio_oferta'].replace(-99, np.nan)

    # Modificacion de Rating a Float (provisional por presencia de datos nulos)
    df_concat['rating'] = df_concat['rating'].replace(np.nan, '-99')
    df_concat['rating'] = df_concat['rating'].astype(float)
    df_concat['rating'] = df_concat['rating'].replace(-99, np.nan)
    
    # Modificacion de fechas corruptas
    df_concat['fecha'] = df_concat['fecha'].str.replace('simpleripleycl/polera-barbados-2000375617528', '2020-12-31T10:00:00')
    df_concat['fecha'] = df_concat['fecha'].str.replace('simpleripleycl/colchon-drimkip-2-pl-2000375241518p', '2020-12-31T10:00:00')
    df_concat['fecha'] = df_concat['fecha'].str.replace('simpleripleycl/pestanas-magneticas-re-utilizables-3-set-mpm00001185754', '2020-12-31T10:00:00')

    # Modificacion a formato datetime para extraccion de meses y años
    df_concat['fecha'] = pd.to_datetime(df_concat['fecha'])
    df_concat['fecha'] = [d.date() for d in df_concat['fecha']]
    df_concat['año'] = pd.DatetimeIndex(df_concat['fecha']).year
    df_concat['mes'] = pd.DatetimeIndex(df_concat['fecha']).month
    
    # Eliminacion de data contaminada en Rating
    df_concat['rating'] = np.where(df_concat['rating']>5, np.nan, df_concat['rating'])
    df_concat['rating'] = df_concat['rating'].replace(0, np.nan)
    
    # Eliminacion de columnas no requeridas
    df_concat = df_concat.drop(columns = ['fecha'])
    
    # Eliminación de registros duplicados en el dataset
    df_concat = df_concat.drop_duplicates() 
    df_concat.reset_index(drop=True, inplace=True)
    
    return df_concat

# 1.3 Funcion de creacion de subset de PARIS

def preproc_paris(dataframe_original):
    """
    crear_subset_paris - Permite crear subsets del dataframe 'paris', a partir de la selección 
    de ciertas columnas del dataframe original, las cuales han sido estblecidas dentro de la presente funcion.

    @parámetros:
        - dataframe_original: Parámetro obligatorio. Set de datos que contiene todos los atributos y/o variables originales,
          correspondiente al tipo pandas.core.frame.DataFrame.

    @salida:
        - Subset de datos.
    """
    dataframe_original['retail'] = 'paris'

    # Homologaciones de categorias
    recod_where(dataframe_original,'categoria',elec_p,"electro")
    recod_where(dataframe_original,'categoria',mujer_p,"mujer")
    recod_where(dataframe_original,'categoria',tecnologia_p,"tecnologia")
    recod_where(dataframe_original,'categoria',computacion_p,"computacion")
    recod_where(dataframe_original,'categoria',gamer_p,"videojuegos")
    recod_where(dataframe_original,'categoria',jugueteria_p,"jugueteria")
    recod_where(dataframe_original,'categoria',hogar_p,"hogar")
    recod_where(dataframe_original,'categoria',moda_p,"moda")
    
    atrib_paris = ['retail', 'codigo', 'producto', 'precio_original', 'url_imagen', 'vendedor', 'url_producto',
          'modelo', 'descripcion', 'comentario_1', 'precio_internet', 'precio_oferta', 'usuario_1', 
          'comentario_2', 'comentario_3', 'comentario_4', 'comentario_5', 'comentario_6', 'comentario_7', 
          'comentario_8', 'comentario_blanco_1', 'comentario_blanco_2', 'rating_1', 'rating_2', 'rating_3',
          'rating_4', 'rating_5', 'rating_6','rating_7', 'rating_8', 'rating_blanco_1', 'rating_blanco_2', 
          'en_blanco', 'en_blanco_2', 'fecha_comentario_1','fecha_comentario_2', 'fecha_comentario_3', 
          'fecha_comentario_4', 'fecha_comentario_5', 'fecha_comentario_6', 'fecha_comentario_7', 
          'fecha_comentario_8', 'en_blanco_3','en_blanco_4', 'en_blanco_5', 'en_blanco_6', 'en_blanco_7', 
          'en_blanco_8', 'en_blanco_9', 'usuario_2', 'usuario_3', 'usuario_4', 'usuario_5', 'usuario_6', 
          'usuario_7','usuario_8', 'en_blanco_10', 'en_blanco_22', 'en_blanco_11','en_blanco_12', 
          'en_blanco_13', 'en_blanco_14', 'en_blanco_15', 'rating_blanco_11', 'rating_blanco_12', 
          'rating_blanco_13', 'rating_blanco_14', 'rating_blanco_15', 'comentario_blanco_11', 
          'comentario_blanco_12','comentario_blanco_13', 'comentario_blanco_14', 'comentario_blanco_15', 
          'precio_2', 'categoria']
                       
    fix_feat = ['retail','categoria', 'codigo', 'producto','precio_original','precio_internet','precio_oferta']
        
    stacks_feat = [['comentario_1', 'rating_1', 'fecha_comentario_1'], ['comentario_2', 'rating_2', 'fecha_comentario_2'],
                   ['comentario_3', 'rating_3', 'fecha_comentario_3'], ['comentario_4', 'rating_4', 'fecha_comentario_4'],
                   ['comentario_5', 'rating_5', 'fecha_comentario_5'], ['comentario_6', 'rating_6', 'fecha_comentario_6'],
                   ['comentario_7', 'rating_7', 'fecha_comentario_7'], ['comentario_8', 'rating_8', 'fecha_comentario_8']]
        
    # Generacion de un dataframe por cada comentario. Posteriormente seran concatenados para formar un unico dataframe
    tmp = [fix_feat+stack for stack in stacks_feat]
    
    list_df = []
    tabla = dict()
    
    for indice, lista in enumerate(tmp):
        tabla['tabla_'+str(indice+1)] = dataframe_original.loc[:, lista]
        list_df.append(tabla['tabla_'+str(indice+1)])
    
    for tabla in list_df:
        tabla.columns = ['retail','categoria','codigo', 'producto', 'precio_original', 'precio_internet', 'precio_oferta','comentario',
                          'rating', 'fecha']
                
    # Concatenacion de dataframes generados
    df_concat = pd.concat(list_df, axis=0)
    
    # Reemplazo de valores vacios por nulos
    nan = ['""','" "','"        "','"         "','"            "','"                      "']
    df_concat = df_concat.replace(nan, np.nan)
    
    # Eliminacion de comillas dobles
    df_concat = df_concat.applymap(lambda x: x.replace('"', '') if type(x) is str else x)
    
    # Eliminacion de puntos
    df_concat = df_concat.applymap(lambda x: x.replace('.', '') if type(x) is str else x)

    # Modificacion de precios a enteros
    df_concat['precio_original'] = df_concat['precio_original'].replace(np.nan, '-99')
    df_concat['precio_original'] = df_concat['precio_original'].astype(float)
    df_concat['precio_original'] = df_concat['precio_original'].replace(-99, np.nan)
    
    df_concat['precio_internet'] = df_concat['precio_internet'].replace(np.nan, '-99')
    df_concat['precio_internet'] = df_concat['precio_internet'].astype(float)
    df_concat['precio_internet'] = df_concat['precio_internet'].replace(-99, np.nan)    
    
    df_concat['precio_oferta'] = df_concat['precio_oferta'].replace(np.nan, '-99')
    df_concat['precio_oferta'] = df_concat['precio_oferta'].astype(float)
    df_concat['precio_oferta'] = df_concat['precio_oferta'].replace(-99, np.nan)

    # Modificacion de Rating a Float (provisional por presencia de datos nulos)
    df_concat['rating'] = df_concat['rating'].replace(np.nan, '-99')
    df_concat['rating'] = df_concat['rating'].astype(float)
    df_concat['rating'] = df_concat['rating'].replace(-99, np.nan)
    
    # Modificacion a formato datetime para extraccion de meses y años
    df_concat['fecha'] = pd.to_datetime(df_concat['fecha'])
    df_concat['fecha'] = [d.date() for d in df_concat['fecha']]
    df_concat['año'] = pd.DatetimeIndex(df_concat['fecha']).year
    df_concat['mes'] = pd.DatetimeIndex(df_concat['fecha']).month

    # Eliminacion de columnas no requeridas
    df_concat = df_concat.drop(columns = ['fecha'])

    # Eliminación de registros duplicados en el dataset
    df_concat = df_concat.drop_duplicates() 
    df_concat.reset_index(drop=True, inplace=True)
    
    return df_concat

# Necesita ser corregida
# 1.3 Funcion creacion de subset global
'''
def crear_subset(dataframe_original, retail):
    if retail == 'lineo':
        salida = crear_subset_lineo(dataframe_original)
    if retail == 'ripley':
        salida = crear_subset_ripley(dataframe_original)
    if retail == 'paris':
        salida = crear_subset_paris(dataframe_original) 
    return salida
'''

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
    
## 2.1 Feature Engineering

sw = ['son', 'aun', 'servicio', 'venia', 'paquete', 'caja','roto','dañado', 'como', 'ademas', 'antes','viene','para',
     'si', 'rapido', 'todo', 'llego', 'antes', 'tiempo', 'entrega', 'despacho', 'poco', 'atrasado', 'tardio', 
     'retrasado', 'destiempo', 'envio', 'pero', 'tiene', 'y', 'ser', 'entregar', 'dañar', 'venir', 'llegar',
      'despachar', 'atrasar', 'tardar', 'retrasar', 'enviar', 'tener', 'ripley', 'paris', 'linio',
      'Linio', 'resenado', 'Resenado', 'falabella', 'Falabella', 'lacoste', 'Lacoste', 'Lineo', 'lineo']
new_stop_words = set(stopwords.words('spanish') + list(punctuation) + sw)

def preprocess_text(text):
    """
    preprocess_text - Permite preprocesar. 

    @parámetros:
        - text: Parámetro obligatorio. Columna que integra texto a preprocesar.

    @salida:
        - Devuelve mismo parámetro de entrada, pero preprocesado mediante expresiones regulares y lematización .
    """
    #Regular expressions
    text = re.sub(r"[^a-zA-Z_]", ' ', text.lower())
    text = text.strip()
    
    # Diccionario
    text = ' '.join([spell.correction(word) for word in text.split(' ')])

    # Lematizacion
    doc = nlp(text)
    text = ' '.join([word.lemma_.lower() for word in doc])
    return text


def prep_cv(data):
    prep =CountVectorizer(analyzer = 'word',
                    preprocessor = preprocess_text,
                    stop_words = new_stop_words, 
                    min_df =2,
                    ngram_range =(1,1),
                    lowercase = True)


    cv_fit = prep.fit_transform(data)
    
    #extraer palabras
    words = prep.get_feature_names()
    words_freq = cv_fit.toarray().sum(axis=0)
    df_words = pd.DataFrame({'words':words, 'frecuencia':words_freq}).sort_values(by='frecuencia', ascending=False)
    return df_words


def visualizar_variables_categoricas(data,variable,year):
    frecuencia_= data[variable].value_counts()
    freq_= pd.DataFrame(frecuencia_).reset_index()
    freq_.columns= [variable, 'frecuencia']

    plt.figure(figsize=(15,7))
    plt.xticks(rotation=35)
    sns.barplot(x= variable, y= 'frecuencia', data = freq_)
    #plt.subplot(2,2, index+1)
    plt.title("Frecuencia de {} de los productos Año: {}".format(variable, year))
    plt.xlabel(variable)
    plt.ylabel("frecuencia")
    plt.show()

def registros_categ_rating(data, year):
    pd.DataFrame({'Categoria': 'belleza', 'Frecuencia': data[data['categoria'] == 'hogar']['rating'].value_counts()})
    list_df = []
    tabla = dict()
    categoria =data['categoria'].value_counts().index.tolist()
    for cat in categoria:
        tabla['tabla_'+str(cat)] = pd.DataFrame({'categoria': cat, 'frecuencia': data[data['categoria'] == cat]['rating'].value_counts()})
        list_df.append(tabla['tabla_'+str(cat)])
    df_concat = pd.concat(list_df, axis=0).reset_index().rename(columns = {'index':'rating'}).sort_values(['rating'])
    
    
    # Se grafica las columnas del df_concat
    sns.set(rc={'axes.facecolor':'peachpuff', 'figure.facecolor':'peachpuff'})
    plt.figure(figsize=(20,10))
    ax = sns.barplot(x="categoria", y="frecuencia", data=df_concat, hue = 'rating', palette="Spectral", saturation = 0.75)
    plt.title(f'Cantidad de Registros por Categoria y Desagregado por Rating - Año: {year}', fontsize ='xx-large')
    plt.xlabel('Categoria de Producto',fontsize='x-large')
    plt.ylabel('Frecuencia',fontsize='x-large')
    plt.legend(fontsize="x-large")
    plt.show()
    
def homologar_categorias(data):
    tecno = ['tecnologia', 'computacion']
    home = ['hogar', 'electro']

    # Homologaciones de atributo categoria
    recod_where(data,'categoria',tecno,"tecnologia")
    recod_where(data,'categoria',home,"hogar")

    # Eliminación de categorias
    data = data.drop(data[data['categoria']=='supermercado'].index)
    data = data.drop(data[data['categoria']=='joyas'].index)
    data = data.drop(data[data['categoria']=='deportes'].index)
    data = data.drop(data[data['categoria']=='jugueteria'].index)
    data = data.drop(data[data['categoria']=='fitness'].index)
    data = data.drop(data[data['categoria']=='libros'].index)
    data=  data.reset_index(drop=True)
    return data

def q_prod_cate(df, year):
    frecuencia_categoria= df['categoria'].value_counts()
    Q_producto_por_categoria= pd.DataFrame(df.groupby('categoria')['producto'].nunique())
    Q_producto_por_categoria.reset_index()
    inputs= pd.DataFrame(frecuencia_categoria).join(Q_producto_por_categoria).reset_index()
    inputs.columns= ['categoria', 'freq_categoria', 'q_prod']
    
    # Se grafica la cantidad de productos por categoria
    plt.figure(figsize=(8,10))
    plt.title(f'Cantidad de productos por frecuencia de categoria - Año: {year}')
    sns.barplot(x= 'freq_categoria', y= 'q_prod', data= inputs, orient= 'h', hue='categoria', palette= 'hls', saturation= 8)
    plt.legend()
       
def coments_by_prod_cat(data):
    catg = data['categoria'].unique()
    unicos = np.asarray([data[data['categoria'] == cat]['producto'].nunique() for cat in data['categoria'].unique()])
    registros = np.asarray([data[data['categoria'] == cat]['producto'].count() for cat in data['categoria'].unique()])
    rat = [round(rate,2) for rate in (registros/unicos)]
    rates = pd.DataFrame({'Categoria':catg, 'comentario por producto':rat}).sort_values('comentario por producto', ascending = False)
    return rates

def histograma(df, year):
    plt.figure(figsize=(10,5))
    plt.title(f'Histograma de la variable precio_internet - Año: {year}')
    sns.histplot(x='precio_internet', data= df)
    
def q_comentarios(df):
    
    rating1 = df['rating'][df.rating == 1]
    rating2 = df['rating'][df.rating == 2]
    rating3 = df['rating'][df.rating == 3]
    rating4 = df['rating'][df.rating == 4]
    rating5 = df['rating'][df.rating == 5]

    total= len(rating1)+ len(rating2)+ len(rating3)+ len(rating4)+ len(rating5)
    print('numero de comentarios con 1 estrella:  {}  {}%'.format(len(rating1),round((len(rating1)/total)*100,2)))
    print('numero de comentarios con 2 estrella:  {}   {}%'.format(len(rating2),round((len(rating2)/total)*100,2)))
    print('numero de comentarios con 3 estrella:  {}   {}%'.format(len(rating3),round((len(rating3)/total)*100,2)))
    print('numero de comentarios con 4 estrella:  {}   {}%'.format(len(rating4),round((len(rating4)/total)*100,2)))
    print('numero de comentarios con 5 estrella:  {}  {}%'.format(len(rating5),round((len(rating5)/total)*100,2)))
    print('Longitud de los datos:                 {}'.format(df.shape[0]))
    
def wordcloud(df):
    # Separo los dataframe por rating
    df_estrella_1= df[df['rating']==1]
    df_estrella_2= df[df['rating']==2]
    df_estrella_3= df[df['rating']==3]
    df_estrella_4= df[df['rating']==4]
    df_estrella_5= df[df['rating']==5]

    # Uno todas las palabras en un string para poder aplicar la libreria de WordCloud

    estrellas_All=" ".join(review for review in df.comentario)
    estrella_1= " ".join(review for review in df.comentario)
    estrella_2 = " ".join(review for review in df.comentario)
    estrella_3 = " ".join(review for review in df.comentario)
    estrella_4 = " ".join(review for review in df.comentario)
    estrella_5 = " ".join(review for review in df.comentario)

    # Se obtiene el objeto listo para crear y generar la nube de palabras:
    wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(estrellas_All)
    wordcloud_1 = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(estrella_1)
    wordcloud_2 = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(estrella_2)
    wordcloud_3 = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(estrella_3)
    wordcloud_4 = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(estrella_4)
    wordcloud_5 = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(estrella_5)


    # Muestra de nube de palabras
    # Gráfico los objetos obtenidos:

    fig, ax = plt.subplots(6, 1, figsize  = (60,60))

    ax[0].imshow(wordcloud_ALL, interpolation='spline16')
    ax[0].set_title('All Estrellas', fontsize=30, color='#1f4068')
    ax[0].axis('off')
    ax[1].imshow(wordcloud_1, interpolation='spline16')
    ax[1].set_title('Frecuencia de palabras en comentarios cuando tienen de rating 1 estrella',fontsize=30, color='#1f4068')
    ax[1].axis('off')
    ax[2].imshow(wordcloud_2, interpolation='spline16')
    ax[2].set_title('Frecuencia de palabras en comentarios cuando tienen de rating 2 estrellas',fontsize=30, color='#1f4068')
    ax[2].axis('off')
    ax[3].imshow(wordcloud_3, interpolation='spline16')
    ax[3].set_title('Frecuencia de palabras en comentarios cuando tienen de rating 3 estrellas',fontsize=30, color='#1f4068')
    ax[3].axis('off')
    ax[4].imshow(wordcloud_4, interpolation='spline16')
    ax[4].set_title('Frecuencia de palabras en comentarios cuando tienen de rating 4 estrellas',fontsize=30, color='#1f4068')
    ax[4].axis('off')
    ax[5].imshow(wordcloud_5, interpolation='spline16')
    ax[5].set_title('Frecuencia de palabras en comentarios cuando tienen de rating 5 estrellas',fontsize=30, color='#1f4068')
    ax[5].axis('off')
    plt.show()
    
def wordcloud_2(df):

    # Uno todas las palabras en un string para poder aplicar la libreria de WordCloud
    comentarios_All="".join(review for review in df.words)


    # Se obtiene el objeto listo para crear y generar la nube de palabras:
    wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="#00263b",colormap='Spectral').generate(comentarios_All)

    # Muestra de nube de palabras
    fig, ax = plt.subplots(1,1 ,figsize  = (60,60))

    ax.imshow(wordcloud_ALL, interpolation='spline16')
    ax.set_title('All Comentarios', fontsize=30, color='#1f4068')
    ax.axis('off')

    plt.show()
    
# FUNCIÓN PARA OBTENER CLASSIFICATION REPORT:
def report(y_train, y_pred_train, y_test, y_pred):
    print("Classification report - TRAIN SET")
    print(classification_report(y_train, y_pred_train))
    print("\nClassification report - TEST SET")
    print(classification_report(y_test, y_pred))

# FUNCIÓN PARA CLASSIFICATION REPORT - SÓLO TEST SET
def test_classification_report(nombre_modelo, y_test, y_hat):
    print("\nTEST SET - Classification report - {}".format(nombre_modelo.upper()) )
    print(classification_report(y_test, y_hat))
    
def dataset_year_filter(dataframe,year):
    """
    dataset_year_filter - Permite depurar el data set - con 9 columnas - en función del año requerido. 

    @parámetros:
        - dataframe: Parámetro obligatorio. Set de datos por analizar, correspondiente al tipo pandas.core.frame.DataFrame.
        - year: Parámetro obligatorio. Año por el cual se busca realizar el filtro de observaciones en el dataset. 

    @salida:
        - Dataframe depurado.
    """
    
    if len(dataframe.columns) == 9:
        
        # Eliminación de columnas que no serán requeridas para generar el dataset depurado
        dataframe_tmp = dataframe.drop(columns = ['retail','codigo','producto'])
        
        if year == 2020:
            # Filtro por año
            dataframe_tmp = dataframe_tmp[dataframe_tmp['año'] == 2020]
            
        if year == 2021:
            dataframe_tmp = dataframe_tmp[dataframe_tmp['año'] == 2021]
        
        # Eliminación de columna año y mes
        dataframe_tmp = dataframe_tmp.drop(columns = ['año','mes'])
        
        # Reseteo del índice del dataframe
        dataframe_tmp = dataframe_tmp.reset_index().drop(columns="index")
        
        return dataframe_tmp

def entrenamiento_modelo_base(modelo, X_train, y_train, nombre_modelo):
    """
    entrenamiento_modelo_base - Permite entrenar un modelo base sin modificación de hiperparámetros, obteniendo su serialización.

    @parámetros:
        - modelo: Parámetro obligatorio. Modelo a implementar, previamente instanciado. 
        - X_train: Parámetro obligatorio. Matriz de atributos de entrenamiento.
        - y_train:Parámetro obligatorio. Vector objetivo de entrenamiento.
        - nombre_modelo: Parámetro obligatorio. Nombre que se asignará al modelo, para identificación. 
    @salida:
        - Modelo serializado.
    """
    pipe = Pipeline(steps = [
    ("tran", tran),
    ("modelo", modelo)])
    
    modelo_tmp = pipe.fit(X_train, y_train)
    
    return modelo_tmp, pickle.dump(modelo_tmp, open(nombre_modelo,"wb"))

def prediccion_reporte_modelo(nombre_modelo, X_train, X_test, y_test):
    """
    prediccion_modelo - Permite realizar las predicciones en base a un modelo previamente serializado, 
    y obtener métricas de desempeño asociadas, tanto para el Train Set como para Test Set. 

    @parámetros:
    - nombre_modelo: Parámetro obligatorio. Nombre de identificación del modelo serializado.
    - X_train: Parámetro obligatorio. Matriz de atributos de entrenamiento.
    - X_test: Parámetro obligatorio. Matriz de atributos para predicciones.
    - y_test:Parámetro obligatorio. Vector objetivo de testeo.
        
    @salida:
        - Reporte de las métricas obtenidas.
    """
    
    # Carga del modelo con pickle
    modelo_tmp = pickle.load(open(nombre_modelo,"rb"))

    # Predicciones para modelo 
    y_pred_train = modelo_tmp.predict(X_train)
    y_pred = modelo_tmp.predict(X_test)
    
    return report(y_train, y_pred_train, y_test, y_pred)