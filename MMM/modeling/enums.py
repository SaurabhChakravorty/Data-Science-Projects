from enum import Enum


class Observed_Type(Enum):
    lc_orders = 'lc_orders'
    lc_revenue = 'lc_revenue'


class Control_Type(Enum):
    trend = 'trend'
    we = 'weekend'
    intercept = 'intercept'
    dummy = 'dummy'
