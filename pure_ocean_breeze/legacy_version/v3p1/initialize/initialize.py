import pickle
import os
import pickledb


def initialize():
    user_file = os.path.expanduser("~") + "/"
    # 日频数据路径
    daily_data_file = input("请设置日频数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    while "/" not in daily_data_file:
        print("请不要输入反斜杠''，请替换为'/'，并以'/'结尾")
        daily_data_file = input("请设置日频数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    if daily_data_file[-1] != "/":
        daily_data_file = daily_data_file + "/"
    # 因子数据路径
    factor_data_file = input("请设置因子数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    while "/" not in factor_data_file:
        print("请不要输入反斜杠''，请替换为'/'，并以'/'结尾")
        factor_data_file = input("请设置因子数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    if factor_data_file[-1] != "/":
        factor_data_file = factor_data_file + "/"
    # 风格数据路径
    barra_data_file = input("请设置barra数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    while "/" not in barra_data_file:
        print("请不要输入反斜杠''，请替换为'/'，并以'/'结尾")
        barra_data_file = input("请设置barra数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    if barra_data_file[-1] != "/":
        barra_data_file = barra_data_file + "/"
    # 更新辅助数据路径
    update_data_file = input("请设置更新辅助数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    while "/" not in update_data_file:
        print("请不要输入反斜杠''，请替换为'/'，并以'/'结尾")
        update_data_file = input("请设置更新辅助数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    if update_data_file[-1] != "/":
        update_data_file = update_data_file + "/"
    # 最终因子成果路径
    final_factor_file = input("请设置最终因子成果存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    while "/" not in final_factor_file:
        print("请不要输入反斜杠''，请替换为'/'，并以'/'结尾")
        final_factor_file = input("请设置最终因子成果存放路径(请最终以斜杠结尾，请不要输入反斜杠'',请都替换为'/')：")
    if final_factor_file[-1] != "/":
        final_factor_file = final_factor_file + "/"
    # 数立方token
    api_token = input("请输入您的数立方token：")
    # 初始化时，日频数据的截止日期
    daily_enddate = input("请输入初始化时（当前）日频数据的截止日期，形如'20220711'：")
    # 初始化时，分钟数据的截止日期
    minute_enddate = input("请输入初始化时（当前）分钟数据的截止日期，形如'20220711'：")
    # 将此结果存入配置文件
    db = pickledb.load(update_data_file + "database_config.db", False)
    db.set("daily_enddate", daily_enddate)
    db.set("minute_enddate", minute_enddate)
    db.dump()
    save_dict = {
        "daily_data_file": daily_data_file,
        "factor_data_file": factor_data_file,
        "barra_data_file": barra_data_file,
        "update_data_file": update_data_file,
        "final_factor_file": final_factor_file,
        "api_token": api_token,
        "daily_enddate": daily_enddate,
    }
    save_dict_file = open(user_file + "paths.settings", "wb")
    pickle.dump(save_dict, save_dict_file)
    save_dict_file.close()
    from loguru import logger

    logger.success("恭喜你，回测框架初始化完成，可以开始使用了👏")
