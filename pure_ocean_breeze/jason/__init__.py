"""
一个量化多因子研究的框架，包含数据、回测、因子加工等方面的功能
"""

__author__ = "chenzongwei"
__author_email__ = "winterwinter999@163.com"
__url__ = "https://github.com/chen-001/pure_ocean_breeze"
__all__ = [
    "data",
    "labor",
    "state",
    "withs"
]

import warnings

warnings.filterwarnings("ignore")
from pure_ocean_breeze.jason.state.homeplace import HomePlace


import pickledb
import datetime

try:
    homeplace = HomePlace()
    db = pickledb.load(homeplace.update_data_file + "use_count.db", False)
    uc_dict = {
        "uc1": "早上",
        "uc2": "上午",
        "uc3": "中午",
        "uc4": "下午",
        "uc5": "傍晚",
        "uc6": "晚上",
        "uc7": "深夜",
    }

    use_count1 = db.get("uc1")
    use_count2 = db.get("uc2")
    use_count3 = db.get("uc3")
    use_count4 = db.get("uc4")
    use_count5 = db.get("uc5")
    use_count6 = db.get("uc6")
    use_count7 = db.get("uc7")



    now = datetime.datetime.now()
    if now.hour >= 6 and now.hour <= 8:
        hello = "早上好"
        use_count1 = use_count1 + 1
    elif now.hour >= 9 and now.hour <= 10:
        hello = "上午好"
        use_count2 = use_count2 + 1
    elif now.hour >= 11 and now.hour <= 12:
        hello = "中午好"
        use_count3 = use_count3 + 1
    elif now.hour >= 13 and now.hour <= 16:
        hello = "下午好"
        use_count4 = use_count4 + 1
    elif now.hour >= 17 and now.hour <= 18:
        hello = "傍晚好"
        use_count5 = use_count5 + 1
    elif now.hour >= 19 and now.hour <= 24:
        hello = "晚上好"
        use_count6 = use_count6 + 1
    else:
        hello = "夜深了"
        use_count7 = use_count7 + 1

    db.set("uc1", int(use_count1))
    db.set("uc2", int(use_count2))
    db.set("uc3", int(use_count3))
    db.set("uc4", int(use_count4))
    db.set("uc5", int(use_count5))
    db.set("uc6", int(use_count6))
    db.set("uc7", int(use_count7))
    db.dump()
    
    def show_use_times():
        db = pickledb.load(homeplace.update_data_file + "use_count.db", False)
        uc = list(db.getall())
        uc = {k: db.get(k) for k in uc}
        con1 = ["在" + uc_dict[k] + "加载" + str(v) + "次" for k, v in uc.items()]
        con1 = "\n".join(con1)
        con1 = "您" + con1
        most = [uc_dict[k] for k, v in uc.items() if v == max(list(uc.values()))]
        con3 = []
        if "深夜" in most:
            con3.append("熬夜伤身，要注意休息哦")
        if "晚上" in most:
            con3.append("晚上要多陪陪家人呀")
        if "早上" in most:
            con3.append("早起小冠军非你莫属啦")
        if "上午" in most:
            con3.append("真是勤奋呀")
        if "下午" in most:
            con3.append("看来大家都喜欢下午工作")
        if "傍晚" in most:
            con3.append("记得趁着黄昏出去看看日落哦")
        if "中午" in most:
            con3.append("正午时分记得午睡哦")

        most = "、".join(most)
        con2 = f"看来您最喜欢在{most}工作"
        con3 = "，".join(con3)
        con = con1 + "\n" + con2 + "\n" + con3
        print(con)
except Exception:
    # print('您可能正在初始化；如果不是在初始化，则路径设置文件已经清除，请检查。')
    ...



from pure_ocean_breeze.jason import state
from pure_ocean_breeze.jason import data
from pure_ocean_breeze.jason import labor


from pure_ocean_breeze.jason.state import *
from pure_ocean_breeze.jason.data import *
from pure_ocean_breeze.jason.labor import *

from pure_ocean_breeze.jason.state.homeplace import *
from pure_ocean_breeze.jason.state.decorators import *

from pure_ocean_breeze.jason.data.dicts import *
from pure_ocean_breeze.jason.data.read_data import *
from pure_ocean_breeze.jason.data.tools import *
from pure_ocean_breeze.jason.data.database import *

from pure_ocean_breeze.jason.labor.process import *
from pure_ocean_breeze.jason.labor.comment import *

import sys
sys.path.append('/home/chenzongwei/pythoncode')


    

    