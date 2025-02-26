import pickle
import os


def ini():
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
    # use all parts
    save_dict = {
        "daily_data_file": daily_data_file,
        "factor_data_file": factor_data_file,
        "barra_data_file": barra_data_file,
    }
    save_dict_file = open(user_file + "paths.settings", "wb")
    pickle.dump(save_dict, save_dict_file)
    save_dict_file.close()
    from loguru import logger

    logger.success("恭喜你，回测框架初始化完成，可以开始使用了👏")
