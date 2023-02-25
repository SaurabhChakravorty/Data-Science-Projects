import random
import string

import pyodbc

import config


def get_cnx():
    return pyodbc.connect(config.Config.db_connection_string)


def get_rnd_chars(length):
    letters = string.ascii_lowercase
    return '(' + ''.join(random.choice(letters) for i in range(length)) + ')'


def dt_to_integer(dt):
    return (
            dt.year * 10000000000 +
            dt.month * 100000000 +
            dt.day * 1000000 +
            dt.hour * 10000 +
            dt.minute * 100 +
            dt.second
    )
