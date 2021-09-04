import os, psycopg2, json, io, base64
from flask.wrappers import Response
from flask import Flask, request, request, jsonify,render_template
#from flask_sqlalchemy import SQLAlchemy
from dboperation import *
# maching learning
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt, mpld3
from flask_cors import CORS, cross_origin
from json import dumps

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")



def init():
    data = selectCategoriasTitulos()
    json_result = json.dumps(data)
    return jsonify(json_result)
    #response = {'content':'Hello World'}
    #return jsonify(response)


@app.route("/clustering", methods = ['GET', 'POST', 'DELETE'])
def clustering ():
    if request.method == 'POST':
        

        body = request.get_json()
        n_clusters      = body["n_clusters"]
        init            = body['init']
        max_iter        = body['max_iter']
        n_init          = body['n_init']
        n_clusters      = body["n_clusters"]
        assign_labels   = body['spectral-assign_labels']     
        eps             = body["dbscan-eps"]
        min_samples     = body['dbscan-min_samples']   
        method          = body["jerarq-method"]         
        affinity        = body['aglo-affinity']
        linkage         = body['aglo-linkage']      

        total_data = selectCategoriasTitulos()

        dataset= pd.DataFrame(total_data, columns=['Categoria', 'Titulo']).reset_index(drop=True)
        X = dataset.apply(LabelEncoder().fit_transform).values

        sc_x = StandardScaler()
        # Se establece una transformacion
        X = sc_x.fit_transform(X)
        #print('fit_transform')        
        #print(X)        

        #/*------------------------Grafico El Metodo del Codo------------------------------------*/
        wcss=[]
        for i in range(1,11): 
            kmeans = KMeans(n_clusters=i, init =init, max_iter=max_iter,n_init=n_init,random_state=0 )
            kmeans.fit(X)            
            wcss.append(kmeans.inertia_)

        plt.plot(range(1,11),wcss)
        plt.title('Grafico El Metodo del Codo')
        plt.xlabel('Numero de clusters')
        plt.ylabel('WCSS')
        plt.show()

        #/* Aplicando Algortimo Kmeans*/
        print('--------------Clustering Kmeans---------------------')
        # kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)    
        y_kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=0).fit_predict(X)
        dataset['Cluster'] = y_kmeans
        #print(dataset)
        #print(dataset.values)        
        print(y_kmeans)        
        #print(kmeans.cluster_centers_)
        #print(kmeans.labels_)        
        #print(kmeans.inertia_)                
        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=50, c='green', label ='Cluster 3')
        plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=50, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=50, c='magenta', label ='Cluster 5')
        #Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label = 'Centroids')
        plt.title('kmeans: Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()

        #/*Clustering Spectral*/
        print('--------------Clustering Spectral---------------------')
        #sc = SpectralClustering(n_clusters=5,assign_labels='discretize',random_state=0).fit(X)
        y_spectral = SpectralClustering(n_clusters=n_clusters,assign_labels=assign_labels,random_state=0).fit_predict(X)   
        dataset['Cluster'] = y_spectral     
        print(y_spectral)

        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_spectral==0, 0], X[y_spectral==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(X[y_spectral==1, 0], X[y_spectral==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(X[y_spectral==2, 0], X[y_spectral==2, 1], s=50, c='green', label ='Cluster 3')
        plt.scatter(X[y_spectral==3, 0], X[y_spectral==3, 1], s=50, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_spectral==4, 0], X[y_spectral==4, 1], s=50, c='magenta', label ='Cluster 5')        
        plt.title('Clustering Spectral: Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()

        #/*DBScan*/
        print('--------------Clustering DBScan---------------------')
        y_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)      
        dataset['Cluster'] = y_dbscan             
        print(y_dbscan)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
        n_noise_    = list(y_dbscan).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)        
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, y_dbscan))
         #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_dbscan==-1, 0], X[y_dbscan==-1, 1], s=50, c='red', label ='Ouliers')
        plt.scatter(X[y_dbscan==0, 0], X[y_dbscan==0, 1], s=50, c='blue', label ='Cluster 1')
        plt.scatter(X[y_dbscan==1, 0], X[y_dbscan==1, 1], s=50, c='green', label ='Cluster 2')
        plt.scatter(X[y_dbscan==2, 0], X[y_dbscan==2, 1], s=50, c='cyan', label ='Cluster 3')
        plt.scatter(X[y_dbscan==3, 0], X[y_dbscan==3, 1], s=50, c='magenta', label ='Cluster 4')        
        plt.title('DBSCAN: Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()  

        #/*Clustering Jerarquico*/
        print('--------------Clustering Jerarquico---------------------')        
        plt.figure(figsize=(10, 7))          
        plt.title('Dendrogramas Clustering Jerarquico: Clusters de Categorias vs Titulos Profesionales')
        dend = shc.dendrogram(shc.linkage(X, method=method))
        print(dend)              
        plt.axhline(y=5, color='r', linestyle='--')
        plt.xlabel('Titulo')
        plt.ylabel('Categoria')
        plt.show()

        #/*Clustering Jerarquico Aglomerativo*/
        print('--------------Clustering Jerarquico Aglomerativo---------------------')
        y_aglomerativo = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage).fit_predict(X)                  
        dataset['Cluster'] = y_aglomerativo   
        print(y_aglomerativo)
        plt.figure(figsize=(10, 7))  
        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_aglomerativo==0, 0], X[y_aglomerativo==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(X[y_aglomerativo==1, 0], X[y_aglomerativo==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(X[y_aglomerativo==2, 0], X[y_aglomerativo==2, 1], s=50, c='green', label ='Cluster 3')
        plt.scatter(X[y_aglomerativo==3, 0], X[y_aglomerativo==3, 1], s=50, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_aglomerativo==4, 0], X[y_aglomerativo==4, 1], s=50, c='magenta', label ='Cluster 5')        
        plt.title('Clustering Jerarquico Aglomerativo: Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()

        #/*--------------------Exponiendo resultados-------------------*/
        #json_result = json.dumps({"prediction": centroids.tolist()})
        #json_result = json.dumps({"prediction": y_kmeans.tolist()})
        #json_result = json.dumps({"prediction": dataset.values.tolist()})
        json_result = jsonify(dataset.values.tolist())

        return  json_result

        
@app.route("/kmeans", methods = ['GET', 'POST', 'DELETE'])
#@cross_origin()
def kmeans():
    if request.method == 'POST':
        body2 = {
            "columns": [
                "titulo_cat",
                "full_descripcion"
            ],
            "n_clusters": 5,
            "init": "k-means++",
            "max_iter": 500,
            "n_init": 10,
            "spectral-assign_labels": "discretize",
            "dbscan-eps": 0.3,
            "dbscan-min_samples": 10,
            "jerarq-method": "ward",
            "aglo-affinity": "euclidean",
            "aglo-linkage": "ward"
        }
        body = request.get_json()
        n_clusters  = body2["n_clusters"]
        init        = body2['init']
        max_iter    = body2['max_iter']
        n_init      = body2['n_init']

        total_data = selectCategoriasTitulos()

        dataset= pd.DataFrame(total_data, columns=['Categoria', 'Titulo']).reset_index(drop=True)
        X = dataset.apply(LabelEncoder().fit_transform).values

        sc_x = StandardScaler()
        # Se establece una transformacion
        X = sc_x.fit_transform(X)          
        
        #/*------------------------Grafico El Metodo del Codo------------------------------------*/
        wcss=[]
        for i in range(1,11): 
            kmeans = KMeans(n_clusters=i, init =init, max_iter=max_iter,n_init=n_init,random_state=0 )
            kmeans.fit(X)            
            wcss.append(kmeans.inertia_)       
        plt.plot(range(1,11),wcss)
        plt.title('Grafico El Metodo del Codo')
        plt.xlabel('Numero de clusters')
        plt.ylabel('WCSS')
        plt.show()

        #/* Aplicando Algortimo Kmeans*/
        print('--------------Clustering KMeans---------------------')
        # kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10, random_state=0)    
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=0)    
        y_kmeans = kmeans.fit_predict(X)            
        dataset['Cluster'] = y_kmeans
        #print(dataset)
        #print(dataset.values)        
        print(y_kmeans)        
        #print(kmeans.cluster_centers_)
        #print(kmeans.labels_)        
        #print(kmeans.inertia_)        

        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=50, c='green', label ='Cluster 3')
        plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=50, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=50, c='magenta', label ='Cluster 5')
        #Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label = 'Centroids')
        plt.title('Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()        

        #json_result = json.dumps({"prediction": centroids.tolist()})
        #json_result = json.dumps({"prediction": y_kmeans.tolist()})
        #json_result = json.dumps({"prediction": dataset.values.tolist()})
        json_result = jsonify(dataset.values.tolist())
        #json_result.headers.add("Access-Control-Allow-Origin", "*")
        return  json_result


@app.route("/spectral", methods = ['GET', 'POST', 'DELETE'])
def spectral():
    if request.method == 'POST':

        body = request.get_json()
        n_clusters      = body["n_clusters"]
        assign_labels   = body['spectral-assign_labels']        

        total_data = selectCategoriasTitulos()

        dataset= pd.DataFrame(total_data, columns=['Categoria', 'Titulo']).reset_index(drop=True)
        X = dataset.apply(LabelEncoder().fit_transform).values

        sc_x = StandardScaler()
        # Se establece una transformacion
        X = sc_x.fit_transform(X)                  

        #/*Clustering Spectral*/
        print('--------------Clustering Spectral---------------------')
        #sc = SpectralClustering(n_clusters=5,assign_labels='discretize',random_state=0).fit(X)
        y_spectral = SpectralClustering(n_clusters=n_clusters,assign_labels=assign_labels,random_state=0).fit_predict(X) 
        dataset['Cluster'] = y_spectral        
        print(y_spectral)
        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_spectral==0, 0], X[y_spectral==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(X[y_spectral==1, 0], X[y_spectral==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(X[y_spectral==2, 0], X[y_spectral==2, 1], s=50, c='green', label ='Cluster 3')
        plt.scatter(X[y_spectral==3, 0], X[y_spectral==3, 1], s=50, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_spectral==4, 0], X[y_spectral==4, 1], s=50, c='magenta', label ='Cluster 5')        
        plt.title('Clustering Spectral: Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()

        json_result = jsonify(dataset.values.tolist())

        #return  json_result

        json_result = json.dumps(total_data)
        return jsonify(json_result)

@app.route("/dbscan", methods = ['GET', 'POST', 'DELETE'])
def dbscan():
    if request.method == 'POST':

        body = request.get_json()
        eps         = body["dbscan-eps"]
        min_samples = body['dbscan-min_samples']        

        total_data = selectCategoriasTitulos()

        dataset= pd.DataFrame(total_data, columns=['Categoria', 'Titulo']).reset_index(drop=True)
        X = dataset.apply(LabelEncoder().fit_transform).values

        sc_x = StandardScaler()
        # Se establece una transformacion
        X = sc_x.fit_transform(X)
        #print('fit_transform')        
        #print(X)                

        #/*DBScan*/
        print('--------------Clustering DBScan---------------------')
        #db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        y_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        dataset['Cluster'] = y_dbscan                
        print(y_dbscan)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
        n_noise_    = list(y_dbscan).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)        
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, y_dbscan))

        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_dbscan==-1, 0], X[y_dbscan==-1, 1], s=50, c='red', label ='Outliers')
        plt.scatter(X[y_dbscan==0, 0], X[y_dbscan==0, 1], s=50, c='blue', label ='Cluster 1')
        plt.scatter(X[y_dbscan==1, 0], X[y_dbscan==1, 1], s=50, c='green', label ='Cluster 2')
        plt.scatter(X[y_dbscan==2, 0], X[y_dbscan==2, 1], s=50, c='cyan', label ='Cluster 3')
        plt.scatter(X[y_dbscan==3, 0], X[y_dbscan==3, 1], s=50, c='magenta', label ='Cluster 4')        
        plt.title('Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()        

        json_result = jsonify(dataset.values.tolist())

        return  json_result

@app.route("/jerarquico", methods = ['GET', 'POST', 'DELETE'])
def jerarquico():
    if request.method == 'POST':     
        body = request.get_json()
        method = body["jerarq-method"]       

        total_data = selectCategoriasTitulos()

        dataset= pd.DataFrame(total_data, columns=['Categoria', 'Titulo']).reset_index(drop=True)
        X = dataset.apply(LabelEncoder().fit_transform).values

        sc_x = StandardScaler()
        # Se establece una transformacion
        X = sc_x.fit_transform(X)                

        #/*Clustering Jerarquico*/
        print('--------------Clustering Jerarquico---------------------')        
        plt.figure(figsize=(10, 7))          
        plt.title('Dendrogramas Clustering Jerarquico: Clusters de Categorias vs Titulos Profesionales')
        dend = shc.dendrogram(shc.linkage(X, method=method)) #method='ward'        
        print(dend)              
        plt.axhline(y=5, color='r', linestyle='--')
        plt.xlabel('Titulo')
        plt.ylabel('Categoria')
        plt.show()       
        
        json_result = jsonify(dataset.values.tolist())

        return  json_result    

@app.route("/aglomerativo", methods = ['GET', 'POST', 'DELETE'])
def aglomerativo():
    if request.method == 'POST':

        body = request.get_json()
        n_clusters = body["n_clusters"]
        affinity   = body['aglo-affinity']
        linkage    = body['aglo-linkage']        

        total_data = selectCategoriasTitulos()

        dataset= pd.DataFrame(total_data, columns=['Categoria', 'Titulo']).reset_index(drop=True)
        X = dataset.apply(LabelEncoder().fit_transform).values

        sc_x = StandardScaler()
        # Se establece una transformacion
        X = sc_x.fit_transform(X)        

        #/*Clustering Jerarquico Aglomerativo*/
        print('--------------Clustering Jerarquico Aglomerativo---------------------') 
        y_aglomerativo = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage).fit_predict(X) #affinity='euclidean', linkage='ward'
        dataset['Cluster'] = y_aglomerativo         
        print(y_aglomerativo)
        plt.figure(figsize=(10, 7))  
        #/*--------------------Visualisando los clusters-------------------*/
        plt.scatter(X[y_aglomerativo==0, 0], X[y_aglomerativo==0, 1], s=50, c='red', label ='Cluster 1')
        plt.scatter(X[y_aglomerativo==1, 0], X[y_aglomerativo==1, 1], s=50, c='blue', label ='Cluster 2')
        plt.scatter(X[y_aglomerativo==2, 0], X[y_aglomerativo==2, 1], s=50, c='green', label ='Cluster 3')
        plt.scatter(X[y_aglomerativo==3, 0], X[y_aglomerativo==3, 1], s=50, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_aglomerativo==4, 0], X[y_aglomerativo==4, 1], s=50, c='magenta', label ='Cluster 5')        
        plt.title('Clustering Jerarquico Aglomerativo: Clusters de Categorias vs Titulos Profesionales')
        plt.xlabel('Categoria')
        plt.ylabel('Titulo')
        plt.show()
        
        json_result = jsonify(dataset.values.tolist())

        return  json_result   

@app.route("/jerarquico_g5", methods = ['GET', 'POST', 'DELETE'])
def jerarquical():
    if request.method == 'POST':
        data = selectCategoriasTitulos()
        json_result  = {}
        return  json_result  


if __name__ == '__main__':
    app.run(debug=True)