'''
初始化时保存的路径
'''

__updated__ = '2022-08-16 15:38:08'

import os
import pickle

class HomePlace(object):
    __slots__ = [
        "daily_data_file",
        "minute_data_file",
        "factor_data_file",
        "barra_data_file",
        "update_data_file",
        "api_token",
        "final_factor_file",
        "daily_enddate",
        "minute_enddate",
    ]

    def __init__(self):
        user_file = os.path.expanduser("~") + "/"
        path_file = open(user_file + "paths.settings", "rb")
        paths = pickle.load(path_file)
        for k in self.__slots__:
            setattr(self, k, paths[k])