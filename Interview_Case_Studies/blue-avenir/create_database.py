import psycopg2
import pandas as pd

hostname  = "localhost"
database  = "Blue_Avenir"
username  = "postgres"
pwd       = "user"
port_id   = "5432"
conn = None
cur  =  None
try:
        conn = psycopg2.connect(
            host = hostname,
            dbname = database,
            user   = username,
            password = pwd ,
            port = port_id
        )
        cur = conn.cursor()
        ## create table TB_AGENTS
        cur.execute('DROP TABLE IF EXISTS TB_AGENTS')
        create_script = '''CREATE TABLE IF NOT EXISTS TB_AGENTS (
                          id serial PRIMARY KEY,
                          version varchar(48) NOT NULL,
                          last_update date)'''
        cur.execute(create_script)
        ## create table TB_EPISODES
        cur.execute('DROP TABLE IF EXISTS TB_EPISODES')
        create_script = '''CREATE TABLE IF NOT EXISTS TB_EPISODES (
                          id serial PRIMARY KEY,
                          customer_id varchar(48) NOT NULL,
                          states varchar(256) NOT NULL,
                          predicted_offer_1 varchar(48) NOT NULL,
                          predicted_offer_2 varchar(48) ,
                          predicted_offer_3 varchar(48) ,
                          last_update date,
                          up_to_date date NOT NULL)'''
        cur.execute(create_script)
        ## create table TB_REWARDS
        cur.execute('DROP TABLE IF EXISTS TB_REWARDS')
        create_script = '''CREATE TABLE IF NOT EXISTS TB_REWARDS (
                          id serial PRIMARY KEY,
                          customer_id varchar(48) NOT NULL,
                          accepted_offer boolean,
                          last_update date)'''
        cur.execute(create_script)
        conn.commit()
        ## create table TB_PRODUCTS_A
        cur.execute('DROP TABLE IF EXISTS TB_PRODUCTS_A')
        create_script = '''CREATE TABLE IF NOT EXISTS TB_PRODUCTS_A (
                            id serial PRIMARY KEY ,
                            offers varchar(48) NOT NULL,
                            name varchar(48) NOT NULL)'''
        cur.execute(create_script)
        conn.commit()
        ## enter the data into TB_PRODUCTS_A
        df = pd.read_csv("company_A_products.csv", delimiter=",")
        # enter this data to DB
        # Open a DB session
        dbSession = psycopg2.connect("dbname='Blue_Avenir' user='postgres' password='user'")
        # Open a database cursor
        dbCursor = dbSession.cursor()
        for i,row in df.iterrows():
            sql = "INSERT INTO TB_PRODUCTS_A (offers, name)" \
                    "VALUES(%s,%s)"
            dbCursor.execute(sql, (row["OFFERS"], row["NAME"]))
            dbSession.commit()
        dbSession.close()
except Exception as Error:
    print(Error)

finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()



