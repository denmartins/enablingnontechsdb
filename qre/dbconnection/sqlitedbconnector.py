import sqlite3
import pandas as pd

class SQLiteDBConnector(object):
    def __init__(self, db_filepath):
        self.connection = self.create_connection(db_filepath)

    def create_connection(self, db_filepath):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        connection = None
        try:
            connection = sqlite3.connect(db_filepath)
        except sqlite3.Error as e:
            raise e
    
        return connection

    def execute_query_to_pandas(self, query):
        result = pd.DataFrame()
        try:
            with self.connection: 
                result = pd.read_sql_query(query, self.connection)
        except pd.io.sql.DatabaseError as e:
            print('------------- Query Error ----------------')
            print(e)
            print('------------------------------------------')
        return result

    def execute_command(self, sql_command):
        result = None
        try:
            cs = self.connection.cursor()
            cs.execute(sql_command)
            result = cs.fetchone()
            cs.close()
        except sqlite3.Error as e:
            raise e 

        return result

    def get_table_names(self):
        table_names = []
        
        with self.connection:
            cs = self.connection.cursor()
            cs.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [t[0] for t in cs.fetchall() if not t[0] in ['sqlite_master', 'global_inverted_index', 'sqlite_sequence']]
            cs.close()
        
        return table_names

    def get_pandas_table_dictionary(self):
        table_dict = dict()
        
        with self.connection:
            cs = self.connection.cursor()
            cs.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [t[0] for t in cs.fetchall()]
            for table in table_names:
                table_dict[table] = pd.read_sql_query(str.format('SELECT * FROM {0}', table), self.connection)
            cs.close()

        return table_dict

    def get_tables(self):
        tables = dict()
        
        table_names = self.get_table_names()
        for tbname in table_names:
            tables[tbname] = self.execute_query_to_pandas(str.format('PRAGMA table_info({0})', tbname))
        
        return tables
    