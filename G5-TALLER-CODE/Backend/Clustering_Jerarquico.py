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

statement = """select o.htitulo_cat,o.htitulo,w.pagina_web,o.empresa,o.lugar,o.salario,date_part('year',o.fecha_publicacion) as periodo,
f_dimPuestoEmpleo(o.id_oferta,7) as funciones,
f_dimPuestoEmpleo(o.id_oferta,1) as conocimiento,
f_dimPuestoEmpleo(o.id_oferta,3) as habilidades,
f_dimPuestoEmpleo(o.id_oferta,2) as competencias,
f_dimPuestoEmpleo(o.id_oferta,17) as certificaciones,
f_dimPuestoEmpleo(o.id_oferta,5) as beneficio,
f_dimPuestoEmpleo(o.id_oferta,11) as formacion
from webscraping w inner join oferta o
on (w.id_webscraping=o.id_webscraping)
where o.id_estado is null;"""


cur.execute(statement)
if(commit): 
    conn.commit()
    
base=cur.fetchall()


base = pd.DataFrame(base)

cur.close()
conn.close()

print(base)

cabeceras = ['Categoría','Titulo','Pagina Web',
'Empresa','Lugar','Salario','Periodo','Funciones',
'Conocimiento','Habilidades','Competencias','Certificaciones',
'Beneficio','Formacion']

base_final = pd.DataFrame()

for i in range (len(cabeceras)):
    ler = pp.LabelEncoder()
    ler.fit(base[i])
    list(ler.classes_)
    nueva_base=ler.transform(base[i])
    base_final[cabeceras[i]] = nueva_base

print(base_final)


clustering_jerarquico = linkage(base_final,'ward')

dendrogram = sch.dendrogram(clustering_jerarquico)


plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")



dendrogram = sch.dendrogram(clustering_jerarquico)

plt.axhline(y=2000, color='r', linestyle='--')

plt.show()



clusters = fcluster(clustering_jerarquico,t=2000,criterion='distance')

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

plt.scatter(base_final[cabeceras[0]],base_final[cabeceras[1]], c=clusters)

plt.show()

