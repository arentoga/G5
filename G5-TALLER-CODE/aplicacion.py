from dbconnection import Database
from preprocessing import delete_empty, remove_accents, remove_punctuation
from settings import *
import pandas as pd


def createDataset(data, parameters=None):
    return pd.DataFrame(data, **parameters)


def clearDataset(dataset, columns):
    
    for name in columns:
        dataset[name] = dataset[name].apply(remove_accents)
        dataset[name] = dataset[name].apply(remove_punctuation)
        dataset[name] = dataset[name].apply(delete_empty)
        
    return dataset


db = Database(dbname=DB_NAME, user=DB_USER, passwd=DB_PASSWORD, host=DB_HOST)
data = db.query("""select distinct o.htitulo_cat,o.htitulo,
                cap.descripcion_normalizada as capacitacion
                from webscraping w inner join oferta o
                on (w.id_webscraping=o.id_webscraping)
                left outer join v_capacitacion cap
                on (o.id_oferta=cap.id_oferta)
                where  o.id_estado is null
                order by 1,2,3;""")

dataset = createDataset(data, {})

dataset = clearDataset(dataset, [2])

db.close()

print(dataset)