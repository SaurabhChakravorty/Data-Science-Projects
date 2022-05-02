import pandas as pd
import psycopg2

def batch_process_company_A(states, offers):

    ## only considering True Offers
    d = {}
    for key, values in offers.items():
        if eval(str(values)) == True:   # for setting boolean condition in string
            d[key] = True

    ## also remove last accepted offer
    last_offer = states["last_offer"]
    if last_offer in d:
        del d[last_offer]

    ## store offers
    if d:
      offers = d

    return offers

def batch_process_company_B(df: pd.DataFrame):
    '''
    Conditions which are met:
    1. Customers in Europe cant have offers 4, 5 and 6
    2. Customers age above 65 can't have OFFERS 1,3 and 4
    :param df: with data
    :return: df with satisfied conditions
    '''
    #print("In process B")
    df.loc[df["region"] == "Europa", ["Offer_4","Offer_5","Offer_6"]] = "False"
    df.loc[df["age"] > 65, ["Offer_1", "Offer_3", "Offer_4"]] = "False"


    return df


def write_to_database(customer_id,states,s,timestamp, last_date):
    # write values to database
    # Open a DB session
    dbSession = psycopg2.connect("dbname='Blue_Avenir' user='postgres' password='user'")
    # Open a database cursor
    dbCursor = dbSession.cursor()
    # Insert statement
    states = str(states)
    offer_1 = s[0]
    offer_2 = s[1]
    offer_3 = s[2]
    sqlInsertRow1 = "INSERT INTO TB_EPISODES (customer_id, states, predicted_offer_1, predicted_offer_2, predicted_offer_3, last_update, up_to_date)" \
                    "VALUES(%s,%s,%s,%s,%s,%s,%s)"
    dbCursor.execute(sqlInsertRow1, (customer_id, states, offer_1, offer_2, offer_3, last_date, timestamp))
    # Commit and close session
    dbSession.commit()
    dbSession.close()

    # done
    print("The following id : {} is written to database".format(customer_id))

def get_offer_name(s:list):
    # read table from database
    # Open a DB session
    dbSession = psycopg2.connect("dbname='Blue_Avenir' user='postgres' password='user'")
    # read the table
    script = "SELECT OFFERS, NAME FROM TB_PRODUCTS_A"
    df = pd.read_sql_query(script, con=dbSession)
    dbSession.close()
    # dict to get reference
    ref = dict(zip(df['offers'], df['name']))

    # dict to get ref, and store
    d = {}
    for i in s:
        d[i] = ref[i]

    return d