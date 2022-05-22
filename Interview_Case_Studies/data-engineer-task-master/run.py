import requests
import pandas as pd


def main():
   '''
    This function will read the JSON file and process it for the status and city name
    The aggregated result will then be returned
   '''
   df = pd.read_json("transactions.json", lines = True, convert_dates = False)
   df['user_status'] = 'paying'
   df['city']        = 'munich'

   for i in range(len(df)):
       user_id = df['user_id'][i]
       date = df['created_at'][i]
       ip = df['ip'][i]

       # get user status
       url = "http://127.0.0.1:8000/" + "user_id/" + str(user_id) + "/date/" + str(date)   # api call
       response = requests.get(url).json()['user_status']
       df['user_status'][i] = response

       # get the city name
       url = "http://127.0.0.1:8000" + "/ip_city/" + ip                                   # api call
       response = requests.get(url).json()['city']
       df['city'][i] = response

   # groupby status and city
   t = pd.DataFrame(df.groupby(['user_status', 'city']).agg({'product_price': 'sum'})).reset_index()
   return t


if __name__ == '__main__':
  df = main()
  print(df)