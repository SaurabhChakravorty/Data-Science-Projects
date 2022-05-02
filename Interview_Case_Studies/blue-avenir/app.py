import uvicorn
import fastapi
import pandas as pd
from RL_output import rl_agent
from fastapi import FastAPI, UploadFile
from run import request_body, request_put
from datetime import datetime
import psycopg2
from batch_process import batch_process_company_A,batch_process_company_B, write_to_database, get_offer_name

# get the FASTAPI method
app = FastAPI()

## First End Point : Method -> GET
@app.get("/is-alive")
def root():
    # write values to database
    # Open a DB session
    dbSession = psycopg2.connect("dbname='Blue_Avenir' user='postgres' password='user'")
    # Open a database cursor
    dbCursor = dbSession.cursor()
    # Get the version
    version = fastapi.__version__
    timestamp = datetime.now()
    sqlInsertRow = "INSERT INTO TB_AGENTS (version, last_update)" \
                    "VALUES(%s,%s)"
    dbCursor.execute(sqlInsertRow, (version, timestamp))
    # Commit and close session
    dbSession.commit()
    dbSession.close()
    return { "status": "I’m alive and running", "agent-version":f"{version}-API"}


## End Point for company A : Method -> POST
@app.post("/predict_company_A")
def predict(data:request_body):
    # data = data.dict()
    customer_id = data.customer_id
    timestamp = data.timestamp
    last_date = datetime.now()
    states = data.states
    offers = data.offers

    # get the prediction after some preprocessing
    offers = batch_process_company_A(states, offers)
    s = rl_agent(customer_state=states, offers=offers)

    # write to database
    write_to_database(customer_id,states,s,timestamp, last_date)

    return { "customer_id": customer_id,
          "response - date": timestamp,
          "best - threeoffers": s
        }

## End Point for company B : Method -> POST
@app.post("/predict_company_B")
def predict(file:UploadFile):
    # read the file and batch process it
    df = pd.read_csv(file.file, delimiter=";")
    df = batch_process_company_B(df)

    # for getting states and offers
    results = pd.DataFrame(columns = ["customer_id", "timestamp", "best - threeoffers"])
    # for all the records
    for f in df.to_dict(orient="records"):
      states = {}
      try:
            customer_id = f['customer_id']
            timestamp   = f['timestamp']
            last_date   = datetime.now()

            ## get states
            states['age'] =    f['age']
            states['region'] = f['region']
            states['gender'] = f['gender']
            states['last_offer'] = f['last_offer']

            # get offers
            offers = dict(zip(list(f.keys())[7:], list(f.values())[7:]))

            # remove all "False" finally as we did for case_1
            offers = batch_process_company_A(states, offers)

            # get the top 3 best offers
            s = rl_agent(customer_state=states, offers=offers)

            # write to database
            write_to_database(customer_id, states, s, timestamp, last_date)

            # append in dataframe to give it to .csv
            results.loc[len(results)] = [customer_id, timestamp, s]

      except :
            print("Cannot process for customer_id : {}".format(customer_id))
            continue
     ## write the .csv
    results.to_csv("Company_B_Results.csv")
    return "The files are processed and following {} records are processed out of {}".format(len(results), len(df))

## End Point for company A : Method -> POST
@app.post("/predict_company_v2")
def predict(data:request_body):
    #data = data.dict()
    customer_id = data.customer_id
    timestamp   = data.timestamp
    last_date   = datetime.now()
    states = data.states
    offers = data.offers

    #get the prediction after some preprocessing
    offers = batch_process_company_A(states, offers)
    s = rl_agent(customer_state=states, offers=offers)

    # store the ref and return
    s_name = get_offer_name(s)

    # write to database
    write_to_database(customer_id, states, s, timestamp, last_date)

    return { "customer_id": customer_id,
          "response - date": timestamp,
          "best - threeoffers": s_name
        }

@app.post("/put")
def put(data:request_put):
    data = data.dict()
    customer_id = data['customer_id']
    timestamp = datetime.now()
    accepted = data['accepted_one_of_the_three_offers']
    if accepted ==  True:
        response = "Success"
    else:
        response = "Failure"
    dbSession = psycopg2.connect("dbname='Blue_Avenir' user='postgres' password='user'")
    # Open a database cursor
    dbCursor = dbSession.cursor()
    sqlInsertRow = "INSERT INTO TB_REWARDS (customer_id,accepted_offer,last_update)" \
                    "VALUES(%s,%s,%s)"
    dbCursor.execute(sqlInsertRow, (customer_id, accepted, timestamp))
    # Commit and close session
    dbSession.commit()
    dbSession.close()
    return {
        "customer_id":customer_id,
        "timestamp" : timestamp,
        "response": response
        }


#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

