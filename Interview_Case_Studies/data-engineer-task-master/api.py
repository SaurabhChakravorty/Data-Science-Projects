import datetime as dt
from fastapi import FastAPI
import uvicorn

class UserStatusSearch:
    P = 'paying'
    C = 'cancelled'
    NP = 'non-paying'

    RECORDS = [
        {'user_id': 1, 'created_at': '2017-01-01T10:00:00', 'status': P},
        {'user_id': 1, 'created_at': '2017-03-01T19:00:00', 'status': P},
        {'user_id': 1, 'created_at': '2017-02-01T12:00:00', 'status': C},
        {'user_id': 2, 'created_at': '2017-09-01T17:00:00', 'status': P},
        {'user_id': 3, 'created_at': '2017-10-01T10:00:00', 'status': P},
        {'user_id': 3, 'created_at': '2016-02-01T05:00:00', 'status': C},
    ]
    def __init__(self):
        self.records = self.RECORDS    # instantiating records
        for i in self.records:         # convert into date time for sorting
            i['created_at'] = dt.datetime.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%S")

    def get_status(self, user_id, date):
        """
        In this function we get the status and simulaneously update our records with the user id and status
        """
        # 1. filter as sort with datetime
        r = list(filter(lambda R: R['user_id'] == user_id, self.records))   # for getting records with user_id
        if len(r) > 0:
            r = sorted(r, key=lambda x: x['created_at']) # sort the elements of date
            # if there is only one element
            i = 0 # iterator
            # 2. If the date is less than first element we label the status as "non paying" as we dont have any information prior to that
            if date < r[i]['created_at'] :
                d = {'user_id': user_id, 'created_at': date, 'status': "non paying"}
                self.records.append(d)
                return "non-paying"
            else:
                i = i + 1

            while i < len(r):
                if date < r[i]['created_at']:
                    # 3. We go into loop and mark the record of the latest element
                    d = {'user_id': user_id, 'created_at': date, 'status':r[i]['status']}
                    self.records.append(d)
                    return r[i]['status']
                i = i + 1
            d = {'user_id': user_id, 'created_at': date, 'status': r[i-1]['status']}
            self.records.append(d)
            # 4. If te date is moretan last element we take the last status
            return r[i-1]['status']

        else:
            d = {'user_id': user_id, 'created_at': date, 'status': 'non-paying'}
            self.records.append(d)
            return "non-paying"


class IpRangeSearch:

    RANGES = {
        'london': [
            {'start': '10.10.0.0', 'end': '10.10.255.255'},
            {'start': '192.168.1.0', 'end': '192.168.1.255'},
        ],
        'munich': [
            {'start': '10.12.0.0', 'end': '10.12.255.255'},
            {'start': '172.16.10.0', 'end': '172.16.11.255'},
            {'start': '192.168.2.0', 'end': '192.168.2.255'},
        ]
    }

    def __init__(self):
        self.ranges = self.RANGES

    def convert_ip(self,ip):
        # convert it into tuple and check the range
        return tuple(int(n) for n in ip.split('.'))

    def check_ipv4_in(self,ip, start, end):
        # check if its in range
        return self.convert_ip(start) <= self.convert_ip(ip) < self.convert_ip(end)

    def get_city(self, ip):
        for i in self.ranges['london']:
            # check for London
            start = i['start']
            end   = i ['end']
            if self.check_ipv4_in(ip, start, end):
                return "London"
        for i in self.ranges['munich']:
            # check for Munich
            start = i['start']
            end   = i['end']
            if self.check_ipv4_in(ip, start, end):
                return "Munich"
        return "unknown"

# get the FASTAPI method
app = FastAPI()
ip_range_search = IpRangeSearch()
user_status_search = UserStatusSearch()


@app.get('/user_id/{user_id}/date/{date}')
async def user_status(user_id, date):
    """Return user status for a given date."""
    user_id = int(user_id)
    date = dt.datetime.strptime(
        date, '%Y-%m-%dT%H:%M:%S')
    return {
        'user_status': user_status_search.get_status(user_id, date)
    }


@app.get('/ip_city/{ip_city}')
async def ip_city(ip_city):
    """Return city for a given ip."""
    return {'city': ip_range_search.get_city(ip_city)}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)
