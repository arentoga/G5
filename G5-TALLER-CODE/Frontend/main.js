

const body2 = {
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
/*
const body2 = {
    "n_clusters": 6,
    "init": "k-means++",
    "max_iter": 300,
    "n_init": 10,
    "random_state": 0
    }
*/
const options = {
    method: 'POST',
    /*headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": true,
        //'Content-Type': 'application/json',
        //'charset': 'utf-8',
    },*/
    
    body: JSON.stringify({ columns: ["titulo_cat", "full_descripcion"], n_clusters: 5, init: "k-means++",max_iter: 500, n_init: 10, 'spectral-assign_labels':"discretize","dbscan-eps":0.3, "dbscan-min_samples": 10,"jerarq-method":"ward", "aglo-affinity":"euclidean", "aglo-linkage":"ward" }),
};


const options2 = {
    method: 'GET',
  

};
const url2 = 'https://delati-pml-back.herokuapp.com/kmeans';
const url22 = 'https://delati-pml-back.herokuapp.com';
const url = 'http://127.0.0.1:5000/kmeans';
const url3 = 'http://127.0.0.1:5000'
const url4 = '/'


//fetch(url,options2)
fetch(url3,options2)
.then(response => response.json())
.then(json => console.log(json))
.catch(err => console.log(err))

/*
fetch(url3, {
	method: 'GET'
}).then(response => response.json())
.then(json => console.log(json))
.catch(err => console.log(err))
*/
/*
fetch(url3, {
	method: 'GET'
}).then(function(response) {
	console.log(response.json())
}).catch(err => console.log(err));
/*
const xhr = new XMLHttpRequest();

function onRequestHandler() {
    if(this.readyState == 4 && this.status ==200){
        console.log(this.response)
    }
    
}

xhr.addEventListener("load",onRequestHandler);
xhr.open("GET",`${url}`,true);
xhr.withCredentials = true;
xhr.send();
*/
