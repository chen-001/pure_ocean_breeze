"""
一个量化多因子研究的框架，包含数据、回测、因子加工等方面的功能
"""

__updated__ = "2023-07-18 14:49:23"
__version__ = "5.0.1"
__author__ = "chenzongwei"
__author_email__ = "winterwinter999@163.com"
__url__ = "https://github.com/chen-001/pure_ocean_breeze"
__all__ = [
    "data",
    "labor",
    "state",
    "mail",
    "initialize",
    "legacy_version",
    "future_version",
    "withs",
]

import requests

try:
    response = requests.get("https://pypi.org/pypi/factor-reader/json", timeout=2)
    latest_version = response.json()["info"]["version"]
    now_version=__version__
    if latest_version!=now_version:
        logger.warning(f'''您当前使用的是{now_version}，最新版本为{latest_version}
            建议您使用`pip install pure_ocean_breeze --upgrade`
            或`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pure_ocean_breeze --upgrade`命令进行更新
            ''')
except Exception:
    ...

import warnings

warnings.filterwarnings("ignore")
from pure_ocean_breeze.state.homeplace import HomePlace
from pure_ocean_breeze.initialize.initialize import *

try:
    homeplace = HomePlace()
    if homeplace.use_full=='yes':
        import os


        def upgrade():
            os.system("pip install pure_ocean_breeze --upgrade -i https://pypi.Python.org/simple/")


        up = upgrade

        from loguru import logger


        import bs4
        from wrapt_timeout_decorator import timeout
        import pickledb
        import datetime


        # check_update()

        from pure_ocean_breeze import state
        from pure_ocean_breeze import data
        from pure_ocean_breeze import labor
        from pure_ocean_breeze import mail
        from pure_ocean_breeze import initialize

        # from pure_ocean_breeze import future_version
        # from pure_ocean_breeze import legacy_version

        from pure_ocean_breeze.state import *
        from pure_ocean_breeze.data import *
        from pure_ocean_breeze.labor import *
        from pure_ocean_breeze.mail import *
        from pure_ocean_breeze.initialize import *

        from pure_ocean_breeze.state.homeplace import *
        from pure_ocean_breeze.state.decorators import *

        from pure_ocean_breeze.data.database import *
        from pure_ocean_breeze.data.dicts import *
        from pure_ocean_breeze.data.read_data import *
        from pure_ocean_breeze.data.tools import *
        from pure_ocean_breeze.data.write_data import *

        from pure_ocean_breeze.labor.process import *
        from pure_ocean_breeze.labor.comment import *

        from pure_ocean_breeze.mail.email import *
except Exception:
    # print('您可能正在初始化；如果不是在初始化，则路径设置文件已经清除，请检查。')
    ...




    

    