# contribuciones: js17

### 📋 Pre-requisitos 
Instalamos las librerias mediante pip
pip install requeriments.txt

### 🔧 Instalación 
Ejecutar los sprints en un entorno de desarrollo
python manage.py runserver


#Link del backend en producción heroku: https://delati-pml-back.herokuapp.com/
#Endpoints:
> get Data :  METHOD: GET https://delati-pml-back.herokuapp.com/
> kmeans : METHOD POST https://delati-pml-back.herokuapp.com/kmeans
          request: { "columns": ["titulo_cat", "full_descripcion"], "n_clusters": 5, "init": "k-means++", "max_iter": 500, "n_init": 10, "spectral-assign_labels":"discretize","dbscan-eps":0.3, "dbscan-min_samples": 10,"jerarq-method":"ward", "aglo-affinity":"euclidean", "aglo-linkage":"ward" }

spectral: METHOD POST https://delati-pml-back.herokuapp.com/spectral
          request: { "columns": ["titulo_cat", "full_descripcion"], "n_clusters": 5, "init": "k-means++", "max_iter": 500, "n_init": 10, "spectral-assign_labels":"discretize","dbscan-eps":0.3, "dbscan-min_samples": 10,"jerarq-method":"ward", "aglo-affinity":"euclidean", "aglo-linkage":"ward" }

dbscan: METHOD POST https://delati-pml-back.herokuapp.com/dbscan
          request: { "columns": ["titulo_cat", "full_descripcion"], "n_clusters": 5, "init": "k-means++", "max_iter": 500, "n_init": 10, "spectral-assign_labels":"discretize","dbscan-eps":0.3, "dbscan-min_samples": 10,"jerarq-method":"ward", "aglo-affinity":"euclidean", "aglo-linkage":"ward" }

jerarquico: METHOD POST https://delati-pml-back.herokuapp.com/jerarquico
          request: { "columns": ["titulo_cat", "full_descripcion"], "n_clusters": 5, "init": "k-means++", "max_iter": 500, "n_init": 10, "spectral-assign_labels":"discretize","dbscan-eps":0.3, "dbscan-min_samples": 10,"jerarq-method":"ward", "aglo-affinity":"euclidean", "aglo-linkage":"ward" }

aglomerativo: METHOD POST https://delati-pml-back.herokuapp.com/aglomerativo
          request: { "columns": ["titulo_cat", "full_descripcion"], "n_clusters": 5, "init": "k-means++", "max_iter": 500, "n_init": 10, "spectral-assign_labels":"discretize","dbscan-eps":0.3, "dbscan-min_samples": 10,"jerarq-method":"ward", "aglo-affinity":"euclidean", "aglo-linkage":"ward" }
