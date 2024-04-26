"""
一个量化多因子研究的框架，包含数据、回测、因子加工等方面的功能
"""

__updated__ = "2023-07-18 14:49:23"
__version__ = "4.2.2"
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

        try:
            
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
            print('您可能正在初始化；如果不是在初始化，则路径设置文件已经清除，请检查。')
            

        @timeout(10)
        def try_check_update():
            res = requests.get("https://pypi.org/project/pure-ocean-breeze/#history").text
            resbs = bs4.BeautifulSoup(res, "html.parser")
            latest_version = (
                resbs.find("p", attrs={"class": "release__version"}).contents[0].strip()
            )

            @timeout(5)
            def try_get_new_updates():
                r = requests.get(
                    "https://github.com/chen-001/pure_ocean_breeze/blob/master/更新日志/version3.md"
                ).text
                rbs = bs4.BeautifulSoup(r, "html.parser")
                rbs.find("ol", attrs={"dir": "auto"})
                new = rbs.find("ol", attrs={"dir": "auto"}).find_all("li")
                new = [str(i + 1) + "." + j.contents[0] for i, j in enumerate(new)]
                new = "\n".join(new)
                new = "\n" + new + "\n"
                new = f"最新版本{latest_version}更新的内容为{new}最近其他"
                return new

            def get_new_updates():
                try:
                    y = try_get_new_updates()
                    return y
                except Exception:
                    return "最近"

            new = get_new_updates()

            if latest_version == __version__:
                logger.success(
                    f"""当前是最新版{latest_version}，请放心使用
        使用中如需要帮助，可以参考说明文档 https://chen-001.github.io/pure_ocean_breeze/ （上方'罗盘'即为各个类和函数说明哦'）"""
                )
            else:
                logger.warning(
                    f"""\n您使用的版本为{__version__}，而当前已经更新至{latest_version}。
        为了避免一些一些潜在的bug或体验上的损失，建议您使用pure_ocean_breeze模块内置函数up或upgrade来升级，您可以使用如下代码
        import pure_ocean_breeze as p
        p.up()
        来升级至最新版，
        或者使用 pip install pure_ocean_breeze --upgrade 命令升级至最新版，
        或者去Pypi官网 https://pypi.org/project/pure-ocean-breeze/#files 下载最新版安装包后，再使用 pip install <文件路径+文件名> 安装。
        {new}版本更新内容等，详见更新日志 https://github.com/chen-001/pure_ocean_breeze/blob/master/更新日志/更新日志.md
        使用中如需要帮助，可以参考说明文档 https://chen-001.github.io/pure_ocean_breeze/  （上方'罗盘'即为各个类和函数说明哦'）
                            """
                )


        def check_update():
            try:
                print(f"👋小可爱/大可爱，{hello}，欢迎使用pure_ocean_breeze回测框架")
                try_check_update()
            except Exception:
                print(f"👋小可爱/大可爱，{hello}，欢迎使用pure_ocean_breeze回测框架，您当前电脑可能已经离线🌙，您也要早点休息哦")


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




    

    