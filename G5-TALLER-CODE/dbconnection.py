import psycopg2

class Database:

    def __init__(self, dbname=None, host=None, port="5432", user=None, passwd=None):
        self.dbname = dbname
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.conn = psycopg2.connect(dbname=self.dbname, host=self.host, port=self.port,
                                   user=self.user, password=self.passwd)
        self.cur =  self.conn.cursor()
        
    def query(self, statement, commit=False):
        self.cur.execute(statement)
        if(commit): 
            self.conn.commit()

        return self.cur.fetchall()

    def close(self):
        self.cur.close()
        self.conn.close()