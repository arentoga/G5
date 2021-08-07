const body = {
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

const options = {
    method: 'POST',
    headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": true,
        'Content-Type': 'application/json',
    },
    mode: 'no-cors',
    body: JSON.stringify(body),
};

const options2 = {
    method: 'GET',
    headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": true,
        'Content-Type': 'application/json',
    },
    mode: 'no-cors',
};

const url = 'https://delati-pml-back.herokuapp.com/'

fetch(url, options2)
    .then(response => response.text())
    .then(data => {
        console.log(data);
    }).catch(err => console.log(err))

