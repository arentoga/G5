# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:59:22 2021

@author: ARTURO
"""

import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import LabelEncoder as le
from sklearn import preprocessing as pp

# DATABASE
DB_NAME = "delati"
DB_HOST = "128.199.1.222"
DB_PORT = "5432"
DB_USER = "modulo4"
DB_PASSWORD = "modulo4"
commit=False



conn = psycopg2.connect(dbname=DB_NAME, host=DB_HOST, port=DB_PORT,
                                   user=DB_USER, password=DB_PASSWORD)
cur =  conn.cursor()

statement = """select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) 
where o.id_estado is null order by 1,2;"""


cur.execute(statement)
if(commit): 
    conn.commit()
    
base=cur.fetchall()


base = pd.DataFrame(base)

cur.close()
conn.close()

print(base)

ler = pp.LabelEncoder()
ler.fit(base[0])
list(ler.classes_)
nueva_base=ler.transform(base[0])
nueva_base

ler = pp.LabelEncoder()
ler.fit(base[1])
list(ler.classes_)
nueva_base_2=ler.transform(base[1])
base_final = pd.DataFrame()
base_final['Titulo'] = nueva_base
base_final['Titulo_O'] = nueva_base_2
base_final

clustering_jerarquico = linkage(base_final,'ward')

dendrogram = sch.dendrogram(clustering_jerarquico)


plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")



dendrogram = sch.dendrogram(clustering_jerarquico)

plt.axhline(y=200, color='r', linestyle='--')

plt.show()



clusters = fcluster(clustering_jerarquico,t=200,criterion='distance')

base['Clustering Jerarquico'] = clusters
base

df = pd.DataFrame(base)
df

cantidad = pd.DataFrame(df.groupby('Clustering Jerarquico').count())
cantidad

porcentaje = cantidad[0]/df.shape[0]*100
porcentaje = porcentaje.round(2) 
porcentaje.astype(str) + '%'


plt.figure(figsize=(10, 7))  

plt.scatter(base_final['Titulo'],base_final['Titulo_O'], c=clusters)

plt.show()

