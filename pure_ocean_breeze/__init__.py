"""
一个量化多因子研究的框架，包含数据、回测、因子加工等方面的功能
"""

__updated__ = '2022-08-28 01:38:33'
__version__ = '3.1.0'
__author__ = 'chenzongwei'
__author_email__ = 'winterwinter999@163.com'
__url__ = 'https://github.com/chen-001/pure_ocean_breeze'
__all__ = ["data", "labor", "state", "mail", "initialize", "legacy_version", "future_version"]

import warnings
warnings.filterwarnings('ignore')

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

from pure_ocean_breeze.state.states import *
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

from pure_ocean_breeze.initialize.initialize import *
