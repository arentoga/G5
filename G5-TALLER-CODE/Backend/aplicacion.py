from dbconnection import Database
from preprocessing import delete_empty, remove_accents,remove_punctuation
from settings import *
import pandas as pd
import psycopg2

# DATABASE
DB_NAME = "delati"
DB_HOST = "128.199.1.222"
DB_PORT = "5432"
DB_USER = "modulo4"
DB_PASSWORD = "modulo4"
commit=False

def createDataset(data, parameters=None):
    return pd.DataFrame(data, **parameters)


def clearDataset(dataset, columns):
    
    for name in columns:
        dataset[name] = dataset[name].apply(remove_accents)
        dataset[name] = dataset[name].apply(remove_punctuation)
        dataset[name] = dataset[name].apply(delete_empty)
    return dataset



conn = psycopg2.connect(dbname=DB_NAME, host=DB_HOST, port=DB_PORT,
                                   user=DB_USER, password=DB_PASSWORD)
cur =  conn.cursor()

statement = """select distinct o.htitulo_cat,o.htitulo,
--vcon.id_oferta,vcon.id_ofertadetalle,
vcap.descripcion_normalizada as capacitacion
from webscraping w inner join oferta o
on (w.id_webscraping=o.id_webscraping)
inner join v_capacitacion vcap
on (o.id_oferta=vcap.id_oferta)
where o.id_estado is null
order by 1,2,3;"""


cur.execute(statement)
if(commit): 
    conn.commit()
    
data=cur.fetchall()


#base = pd.DataFrame(base)

cur.close()
conn.close()


dataset = createDataset(data, {})

#dataset = clearDataset(dataset, [2])

print(dataset)