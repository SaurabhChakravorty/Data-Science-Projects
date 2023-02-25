import sys

from singleton_pattern import Singleton

"""
To dos:
"""


@Singleton
class PlatformManager:
    def __init__(self):
        if sys.platform.startswith('linux'):
            self.stack_size = 5000000000
            self.recursion_limit = 100000
            self.prior_samples = 10000
            self.driver = "{ODBC Driver 17 for SQL Server}"
        else:
            self.stack_size = 4000000
            self.recursion_limit = 100000
            self.prior_samples = 100
            self.driver = "{SQL Server Native Client 11.0}"
