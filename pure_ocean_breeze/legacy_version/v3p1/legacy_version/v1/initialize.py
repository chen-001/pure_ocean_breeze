import pickle
import os

def initialize():
    # print('正在安装依赖库，请稍等')
    # os.system('pip install numpy')
    # os.system('pip install pandas')
    # os.system('pip install scipy')
    # os.system('pip install statsmodels')
    # os.system('pip install plotly')
    # os.system('pip install matplotlib')
    # os.system('pip install loguru')
    # os.system('pip install h5py')
    # os.system('pip install cachier')
    user_file=os.path.expanduser('~')+'/'
    daily_data_file=input("请设置日频数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    while '/' not in daily_data_file:
        print("请不要输入反斜杠'\'，请替换为'/'，并以'/'结尾")
        daily_data_file=input("请设置日频数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    minute_data_file=input("请设置分钟数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    while '/' not in minute_data_file:
        print("请不要输入反斜杠'\'，请替换为'/'，并以'/'结尾")
        minute_data_file=input("请设置分钟数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    factor_data_file=input("请设置因子数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    while '/' not in factor_data_file:
        print("请不要输入反斜杠'\'，请替换为'/'，并以'/'结尾")
        factor_data_file=input("请设置因子数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    barra_data_file=input("请设置barra数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    while '/' not in barra_data_file:
        print("请不要输入反斜杠'\'，请替换为'/'，并以'/'结尾")
        barra_data_file=input("请设置barra数据存放路径(请最终以斜杠结尾，请不要输入反斜杠'\',请都替换为'/')：")
    save_dict={'daily_data_file':daily_data_file,'minute_data_file':minute_data_file,
               'factor_data_file':factor_data_file,'barra_data_file':barra_data_file}
    save_dict_file=open(user_file+'paths.settings','wb')
    pickle.dump(save_dict,save_dict_file)
    save_dict_file.close()
    from loguru import logger
    logger.success('恭喜你，回测框架初始化完成，可以开始使用了👏')
