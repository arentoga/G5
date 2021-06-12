
import psycopg2
import businessmain
from database import Connection
import sys
from nltk.corpus import stopwords

def connect_bd():
    con = Connection("128.199.1.222", "5432", "delati", "modulo4", "modulo4")
    con.connect()
    return con

if __name__ == "__main__":
    con = connect_bd()
    listaRequisitos = con.select_requisitos()
    businessmain.registrar_normalizado(con, listaRequisitos)