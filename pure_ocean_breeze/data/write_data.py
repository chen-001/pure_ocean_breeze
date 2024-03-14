__updated__ = "2023-08-07 10:47:39"

import time

try:
    import rqdatac

    rqdatac.init()
except Exception as e:
    print(e)
    desicion = input("米筐连接暂时有错误，是否等待并继续连接，等待请输入y，不等待则输入n")
    if desicion == "y":
        print("连接米筐暂时有错误，将等待30秒后重试")
        time.sleep(30)
        try:
            import rqdatac

            rqdatac.init()
        except Exception as e:
            print(e)
            print("连接米筐暂时有错误，将等待60秒后重试")
            time.sleep(60)
            try:
                import rqdatac

                rqdatac.init()
            except Exception as e:
                print(e)
                print("连接米筐暂时有错误，将等待60秒后重试")
                time.sleep(60)
                try:
                    import rqdatac

                    rqdatac.init()
                except Exception as e:
                    print(e)
                    print("暂时未连接米筐")
    else:
        print("暂时未连接米筐")

from loguru import logger
import os
import time
import datetime
import numpy as np
import pandas as pd
import psycopg2 as pg
import scipy.io as scio
from sqlalchemy import FLOAT, INT, VARCHAR, BIGINT
from tenacity import retry
import pickledb
import tqdm.auto
from functools import reduce
from typing import Union, List
import dcube as dc
import py7zr
import unrar
import zipfile
import rarfile
import shutil
import chardet
from tenacity import retry, stop_after_attempt
import questdb.ingress as qdbing
from pure_ocean_breeze.state.homeplace import HomePlace

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")
try:
    pro = dc.pro_api(homeplace.api_token)
except Exception:
    print("暂时未连接数立方")
from pure_ocean_breeze.data.database import (
    sqlConfig,
    ClickHouseClient,
    PostgreSQL,
    Questdb,
)
from pure_ocean_breeze.data.read_data import (
    read_daily,
    read_money_flow,
    read_zxindustry_prices,
    read_swindustry_prices,
    get_industry_dummies,
)
from pure_ocean_breeze.data.dicts import INDUS_DICT, INDEX_DICT, ZXINDUS_DICT
from pure_ocean_breeze.data.tools import (
    生成每日分类表,
    add_suffix,
    convert_code,
    drop_duplicates_index,
    select_max,
)
from pure_ocean_breeze.labor.process import pure_fama



# 待补充
def database_update_second_data_to_clickhouse():
    ...
    
    
    
def convert_tick_by_tick_data_to_parquet(file_name:str,PATH:str,delete_7z:bool=False):
    try:
        files = sorted(os.listdir(file_name))
        files=[i for i in files if i[0]!='.']
        files = [file_name + "/" + i for i in files]
        dfs = []
        for i in files:
            with open(i,'rb') as f:
                tmp = chardet.detect(f.read())
            df = pd.read_csv(i,encoding=tmp['encoding'])
            df.Time = file_name.split('/')[-1] + " " + df.Time
            df.Time = pd.to_datetime(df.Time)
            df = df.rename(
                columns={
                    "TranID": "tranid",
                    "Time": "date",
                    "Price": "price",
                    "Volume": "amount",
                    "SaleOrderVolume": "saleamount",
                    "BuyOrderVolume": "buyamount",
                    "Type": "action",
                    "SaleOrderID": "saleid",
                    "SaleOrderPrice": "saleprice",
                    "BuyOrderID": "buyid",
                    "BuyOrderPrice": "buyprice",
                }
            )
            df = df.assign(code=add_suffix(i.split("/")[-1].split(".")[0]))
            dfs.append(df)
        dfs = pd.concat(dfs)
        dfs.to_parquet(f"{'/'.join(PATH.split('/')[:-2])}/data/{file_name.split('/')[-1]}.parquet")
        # logger.success(f"{file_name.split('/')[-1]}的逐笔数据已经写入完成！")
        shutil.rmtree(file_name + "/",True)
        if delete_7z:
            os.remove(file_name + ".7z")
            
        # logger.warning(f"{file_name.split('/')[-1]}的逐笔数据csv版已经删除")
    except Exception:
        file_name=file_name+'/'+file_name.split('/')[-1]
        convert_tick_by_tick_data_to_parquet(file_name,PATH)
        
        
def convert_tick_by_tick_data_daily(day_path:str,PATH:str):
    try:
        olds=os.listdir('/Volumes/My Passport/data/')
        theday=day_path.split('/')[-1].split('.')[0]
        olds_ok=[i for i in olds if theday in i]
        if len(olds_ok)==0:
            if os.path.exists(day_path.split('.')[0]):
                ...
                # print(f"{day_path.split('.')[0]}已存在")
            elif day_path.endswith('.7z'):
                archive = py7zr.SevenZipFile(day_path, mode='r')
                archive.extractall(path='/'.join(day_path.split('/')[:-1]))
                archive.close()
            elif day_path.endswith('.zip'):
                f = zipfile.ZipFile(day_path,'r') # 压缩文件位置
                f.extractall('/'.join(day_path.split('/')[:-1]))             # 解压位置
                f.close()
            elif day_path.endswith('.rar'):
                f=rarfile.RarFile(day_path,'r')
                f.extractall('/'.join(day_path.split('/')[:-1]))
                f.close()
            convert_tick_by_tick_data_to_parquet(day_path.split('.')[0],PATH)
        else:
            print(f'{theday}已经有了，跳过')
    except Exception:
        logger.error(f'{day_path}出错了，请当心！！！')
        

def convert_tick_by_tick_data_monthly(month_path:str,PATH:str):
    files=os.listdir(month_path)
    files=[i for i in files if i.startswith('20')]
    date=month_path.split('/')[-1]
    files=[month_path+'/'+i for i in files] # 每个形如2018-01-02.7z
    for i in tqdm.auto.tqdm(files,f'{date}的进度'):
        convert_tick_by_tick_data_daily(i,PATH)
        
        
def database_update_minute_data_to_clickhouse_and_questdb(
    kind: str, web_port: str = "9001"
) -> None:
    """使用米筐更新分钟数据至clickhouse和questdb中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'
    web_port : str
        questdb数据库的web console的端口号, by default '9001'

    Raises
    ------
    `IOError`
        如果未指定股票还是指数，将报错
    """
    if kind == "stock":
        code_type = "CS"
    elif kind == "index":
        code_type = "INDX"
    else:
        raise IOError("总得指定一种类型吧？请从stock和index中选一个")
    # 获取剩余使用额
    user1 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    logger.info(f"今日已使用rqsdk流量{user1}MB")
    # 获取全部股票/指数代码
    cs = rqdatac.all_instruments(type=code_type, market="cn", date=None)
    codes = list(cs.order_book_id)
    # 获取上次更新截止时间
    chc = ClickHouseClient("minute_data")
    last_date = max(chc.show_all_dates(f"minute_data_{kind}"))
    # 本次更新起始日期
    start_date = pd.Timestamp(str(last_date)) + pd.Timedelta(days=1)
    start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
    # 本次更新终止日期
    end_date = datetime.datetime.now()
    if end_date.hour < 17:
        end_date = end_date - pd.Timedelta(days=1)
    end_date = datetime.datetime.strftime(end_date, "%Y-%m-%d")
    logger.info(f"本次将下载从{start_date}到{end_date}的数据")
    # 下载数据
    ts = rqdatac.get_price(
        codes,
        start_date=start_date,
        end_date=end_date,
        frequency="1m",
        fields=["volume", "total_turnover", "high", "low", "close", "open"],
        adjust_type="none",
        skip_suspended=False,
        market="cn",
        expect_df=True,
    )
    if ts is not None:
        # 调整数据格式
        ts = ts.reset_index()
        ts = ts.rename(
            columns={
                "order_book_id": "code",
                "datetime": "date",
                "volume": "amount",
                "total_turnover": "money",
            }
        )
        ts = ts.sort_values(["code", "date"])
        ts.date = ts.date.dt.strftime("%Y%m%d").astype(int)
        ts = ts.groupby(["code", "date"]).apply(
            lambda x: x.assign(num=list(range(1, x.shape[0] + 1)))
        )
        ts = (np.around(ts.set_index("code"), 2) * 100).astype(int).reset_index()
        ts.code = ts.code.str.replace(".XSHE", ".SZ")
        ts.code = ts.code.str.replace(".XSHG", ".SH")
        # 数据写入数据库
        ts.to_sql(f"minute_data_{kind}", chc.engine, if_exists="append", index=False)
        ts = ts.set_index("code")
        ts = ts / 100
        ts = ts.reset_index()
        ts.date = ts.date.astype(int).astype(str)
        ts.num = ts.num.astype(int).astype(str)
        qdb = Questdb(web_port=web_port)
        qdb.write_via_df(ts, f"minute_data_{kind}")
        # 获取剩余使用额
        user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
        user12 = round(user2 - user1, 2)
        logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")
    else:
        logger.warning(f"从{start_date}到{end_date}暂无数据")


def database_update_minute_data_to_questdb(kind: str, web_port: str = "9001") -> None:
    """使用米筐更新分钟数据至questdb中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'
    web_port : str
        questdb数据库的控制台端口号, by default '9001'

    Raises
    ------
    `IOError`
        如果未指定股票还是指数，将报错
    """
    if kind == "stock":
        code_type = "CS"
    elif kind == "index":
        code_type = "INDX"
    else:
        raise IOError("总得指定一种类型吧？请从stock和index中选一个")
    # 获取剩余使用额
    user1 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    logger.info(f"今日已使用rqsdk流量{user1}MB")
    # 获取全部股票/指数代码
    cs = rqdatac.all_instruments(type=code_type, market="cn", date=None)
    codes = list(cs.order_book_id)
    # 获取上次更新截止时间
    qdb = Questdb(web_port=web_port)
    last_date = max(qdb.show_all_dates(f"minute_data_{kind}"))
    # 本次更新起始日期
    start_date = pd.Timestamp(str(last_date)) + pd.Timedelta(days=1)
    start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
    # 本次更新终止日期
    end_date = datetime.datetime.now()
    if end_date.hour < 17:
        end_date = end_date - pd.Timedelta(days=1)
    end_date = datetime.datetime.strftime(end_date, "%Y-%m-%d")
    logger.info(f"本次将下载从{start_date}到{end_date}的数据")
    # 下载数据
    ts = rqdatac.get_price(
        codes,
        start_date=start_date,
        end_date=end_date,
        frequency="1m",
        fields=["volume", "total_turnover", "high", "low", "close", "open"],
        adjust_type="none",
        skip_suspended=False,
        market="cn",
        expect_df=True,
    )
    if ts is not None:
        # 调整数据格式
        ts = ts.reset_index()
        ts = ts.rename(
            columns={
                "order_book_id": "code",
                "datetime": "date",
                "volume": "amount",
                "total_turnover": "money",
            }
        )
        ts = ts.sort_values(["code", "date"])
        ts.date = ts.date.dt.strftime("%Y%m%d").astype(int)
        ts = ts.groupby(["code", "date"]).apply(
            lambda x: x.assign(num=list(range(1, x.shape[0] + 1)))
        )
        ts = ts.ffill().dropna()
        ts.code = ts.code.str.replace(".XSHE", ".SZ")
        ts.code = ts.code.str.replace(".XSHG", ".SH")
        ts.date = ts.date.astype(int).astype(str)
        ts.num = ts.num.astype(int).astype(str)
        # 数据写入数据库
        qdb.write_via_df(ts, f"minute_data_{kind}")
        # 获取剩余使用额
        user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
        user12 = round(user2 - user1, 2)
        logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")
    else:
        logger.warning(f"从{start_date}到{end_date}暂无数据")


@retry
def download_single_daily(day):
    """更新单日的数据"""
    try:
        # 8个价格，交易状态，成交量，
        df1 = pro.a_daily(
            trade_date=day,
            fields=[
                "code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjopen",
                "adjclose",
                "adjhigh",
                "adjlow",
                "tradestatus",
                "adjfactor",
                "limit",
                "stopping",
                "avgprice",
            ],
        )
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2 = pro.daily_basic(
            trade_date=day,
            fields=[
                "ts_code",
                "trade_date",
                "turnover_rate",
                "total_share",
                "float_share",
                "pe",
                "pb",
                'pe_ttm',
            ],
        )
        time.sleep(1)
        return df1, df2
    except Exception:
        time.sleep(60)
        # 8个价格，交易状态，成交量，
        df1 = pro.a_daily(
            trade_date=day,
            fields=[
                "code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjopen",
                "adjclose",
                "adjhigh",
                "adjlow",
                "tradestatus",
                "adjfactor",
                "limit",
                "stopping",
                "avgprice",
            ],
        )
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2 = pro.daily_basic(
            trade_date=day,
            fields=[
                "ts_code",
                "trade_date",
                "turnover_rate",
                "total_share",
                "float_share",
                "pe",
                "pb",
                'pe_ttm'
            ],
        )
        time.sleep(1)
        return df1, df2


@retry
def download_calendar(startdate, enddate):
    """更新单日的数据"""
    try:
        # 交易日历
        df0 = pro.a_calendar(start_date=startdate, end_date=enddate)
        time.sleep(1)
        return df0
    except Exception:
        time.sleep(60)
        # 交易日历
        df0 = pro.a_calendar(start_date=startdate, end_date=enddate)
        time.sleep(1)
        return df0


def database_update_daily_files() -> None:
    """更新数据库中的日频数据

    Raises
    ------
    `ValueError`
        如果上次更新到本次更新没有新的交易日，将报错
    """
    homeplace = HomePlace()

    def single_file(name):
        df = pd.read_parquet(homeplace.daily_data_file + name + ".parquet")
        startdate = df.index.max() + pd.Timedelta(days=1)
        return startdate

    names = [
        "opens",
        "highs",
        "lows",
        "closes",
        "trs",
        "opens_unadj",
        "highs_unadj",
        "lows_unadj",
        "closes_unadj",
        "sharenums",
        "total_sharenums",
        "ages",
        "sts",
        "states",
        "amounts",
        "pb",
        "pe",
        'pettm',
        "vwaps",
        "adjfactors",
        "stop_ups",
        "stop_downs",
    ]
    startdates = list(map(single_file, names))
    startdate = min(startdates)
    startdate = datetime.datetime.strftime(startdate, "%Y%m%d")
    now = datetime.datetime.now()
    if now.hour < 17:
        now = now - pd.Timedelta(days=1)
    now = datetime.datetime.strftime(now, "%Y%m%d")
    logger.info(f"日频数据上次更新到{startdate},本次将更新到{now}")

    # 交易日历
    df0 = download_calendar(startdate, now)
    tradedates = sorted(list(set(df0.trade_date)))
    finish = 1
    if len(tradedates) > 1:
        # 存储每天数据
        df1s = []
        df2s = []
        for day in tqdm.auto.tqdm(tradedates, desc="正在下载日频数据"):
            df1, df2 = download_single_daily(day)
            df1s.append(df1)
            df2s.append(df2)
        # 8个价格，交易状态，成交量，
        df1s = pd.concat(df1s)
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2s = pd.concat(df2s)
    elif len(tradedates) == 1:
        df1s, df2s = download_single_daily(tradedates[0])
    else:
        finish = 0
        logger.info("从上次更新到这次更新，还没有经过交易日。放假就好好休息吧，别跑代码了🤒")
    if finish:
        df1s.tradestatus = (df1s.tradestatus == "交易") + 0
        df2s = df2s.rename(columns={"ts_code": "code"})
        df1s.trade_date = pd.to_datetime(df1s.trade_date, format="%Y%m%d")
        df2s.trade_date = pd.to_datetime(df2s.trade_date, format="%Y%m%d")
        df1s = df1s.rename(columns={"trade_date": "date"})
        df2s = df2s.rename(columns={"trade_date": "date"})
        both_codes = list(set(df1s.code) & set(df2s.code))
        df1s = df1s[df1s.code.isin(both_codes)]
        df2s = df2s[df2s.code.isin(both_codes)]
        # st股
        df3 = pro.ashare_st()

        def to_mat(df, row, name, ind="date", col="code"):
            df = df[[ind, col, row]].pivot(index=ind, columns=col, values=row)
            old = pd.read_parquet(homeplace.daily_data_file + name + ".parquet")
            new = pd.concat([old, df]).drop_duplicates()
            new = drop_duplicates_index(new)
            new = new[sorted(list(new.columns))]
            new.to_parquet(homeplace.daily_data_file + name + ".parquet")
            logger.success(name + "已更新")
            return new

        # 股票日行情（未复权高开低收，复权高开低收，交易状态，成交量）
        part1 = df1s.copy()
        part1.volume = part1.volume * 100
        # 未复权开盘价
        opens = to_mat(part1, "open", "opens_unadj")
        # 未复权最高价
        highs = to_mat(part1, "high", "highs_unadj")
        # 未复权最低价
        lows = to_mat(part1, "low", "lows_unadj")
        # 未复权收盘价
        closes = to_mat(part1, "close", "closes_unadj")
        # 成交量
        volumes = to_mat(part1, "volume", "amounts")
        # 复权开盘价
        diopens = to_mat(part1, "adjopen", "opens")
        # 复权最高价
        dihighs = to_mat(part1, "adjhigh", "highs")
        # 复权最低价
        dilows = to_mat(part1, "adjlow", "lows")
        # 复权收盘价
        dicloses = to_mat(part1, "adjclose", "closes")
        # 交易状态
        status = to_mat(part1, "tradestatus", "states")
        # 平均价格
        vwaps = to_mat(part1, "avgprice", "vwaps")
        # 复权因子
        adjfactors = to_mat(part1, "adjfactor", "adjfactors")
        # 涨停价
        stop_ups = to_mat(part1, "limit", "stop_ups")
        # 跌停价
        stop_downs = to_mat(part1, "stopping", "stop_downs")

        # 换手率
        part2 = df2s[["date", "code", "turnover_rate"]].pivot(
            index="date", columns="code", values="turnover_rate"
        )
        part2 = part2 / 100
        part2_old = pd.read_parquet(homeplace.daily_data_file + "trs.parquet")
        part2_new = pd.concat([part2_old, part2])
        part2_new = part2_new.drop_duplicates()
        part2_new = part2_new[closes.columns]
        part2_new = part2_new[sorted(list(part2_new.columns))]
        part2_new = drop_duplicates_index(part2_new)
        part2_new.to_parquet(homeplace.daily_data_file + "trs.parquet")
        logger.success("换手率更新完成")

        # 流通股数
        # 读取新的流通股变动数
        part3 = df2s[["date", "code", "float_share"]].pivot(
            columns="code", index="date", values="float_share"
        )
        part3 = part3 * 10000
        part3_old = pd.read_parquet(homeplace.daily_data_file + "sharenums.parquet")
        part3_new = pd.concat([part3_old, part3]).drop_duplicates()
        part3_new = part3_new[closes.columns]
        part3_new = drop_duplicates_index(part3_new)
        part3_new = part3_new[sorted(list(part3_new.columns))]
        part3_new.to_parquet(homeplace.daily_data_file + "sharenums.parquet")
        logger.success("流通股数更新完成")

        # 总股数
        # 读取新的总股变动数
        part3a = df2s[["date", "code", "total_share"]].pivot(
            columns="code", index="date", values="total_share"
        )
        part3a = part3a * 10000
        part3_olda = pd.read_parquet(
            homeplace.daily_data_file + "total_sharenums.parquet"
        )
        part3_newa = pd.concat([part3_olda, part3a]).drop_duplicates()
        part3_newa = part3_newa.reindex(columns=closes.columns)
        part3_newa = drop_duplicates_index(part3_newa)
        part3_newa = part3_newa[sorted(list(part3_newa.columns))]
        part3_newa.to_parquet(homeplace.daily_data_file + "total_sharenums.parquet")
        logger.success("总股数更新完成")

        # pb
        partpb = df2s[["date", "code", "pb"]].pivot(
            index="date", columns="code", values="pb"
        )
        partpb_old = pd.read_parquet(homeplace.daily_data_file + "pb.parquet")
        partpb_new = pd.concat([partpb_old, partpb])
        partpb_new = partpb_new.drop_duplicates()
        partpb_new = partpb_new[closes.columns]
        partpb_new = partpb_new[sorted(list(partpb_new.columns))]
        partpb_new = drop_duplicates_index(partpb_new)
        partpb_new.to_parquet(homeplace.daily_data_file + "pb.parquet")
        logger.success("市净率更新完成")

        # pe
        partpe = df2s[["date", "code", "pe"]].pivot(
            index="date", columns="code", values="pe"
        )
        partpe_old = pd.read_parquet(homeplace.daily_data_file + "pe.parquet")
        partpe_new = pd.concat([partpe_old, partpe])
        partpe_new = partpe_new.drop_duplicates()
        partpe_new = partpe_new[closes.columns]
        partpe_new = partpe_new[sorted(list(partpe_new.columns))]
        partpe_new = drop_duplicates_index(partpe_new)
        partpe_new.to_parquet(homeplace.daily_data_file + "pe.parquet")
        logger.success("市盈率更新完成")
        
        # pettm
        partpe = df2s[["date", "code", "pe_ttm"]].pivot(
            index="date", columns="code", values="pe_ttm"
        )
        partpe_old = pd.read_parquet(homeplace.daily_data_file + "pettm.parquet")
        partpe_new = pd.concat([partpe_old, partpe])
        partpe_new = partpe_new.drop_duplicates()
        partpe_new = partpe_new[closes.columns]
        partpe_new = partpe_new[sorted(list(partpe_new.columns))]
        partpe_new = drop_duplicates_index(partpe_new)
        partpe_new.to_parquet(homeplace.daily_data_file + "pettm.parquet")
        logger.success("TTM市盈率更新完成")

        # st
        part4 = df3[["s_info_windcode", "entry_dt", "remove_dt"]]
        part4 = part4.sort_values("s_info_windcode")
        part4.remove_dt = part4.remove_dt.fillna(now).astype(int)
        part4 = part4.set_index("s_info_windcode").stack()
        part4 = part4.reset_index().assign(
            he=sorted(list(range(int(part4.shape[0] / 2))) * 2)
        )
        part4 = part4.drop(columns=["level_1"])
        part4.columns = ["code", "date", "he"]
        part4.date = pd.to_datetime(part4.date, format="%Y%m%d")

        def single(df):
            full = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())})
            df = pd.merge(full, df, on=["date"], how="left")
            df = df.fillna(method="ffill")
            return df

        tqdm.auto.tqdm.pandas()
        part4 = part4.groupby(["code", "he"]).progress_apply(single)
        part4 = part4[part4.date.isin(list(part2_new.index))]
        part4 = part4.reset_index(drop=True)
        part4 = part4.assign(st=1)

        part4 = part4.drop_duplicates(subset=["date", "code"]).pivot(
            index="date", columns="code", values="st"
        )

        part4_0 = pd.DataFrame(0, columns=part2_new.columns, index=part2_new.index)
        part4_0 = part4_0 + part4
        part4_0 = part4_0.replace(np.nan, 0)
        part4_0 = part4_0[part4_0.index.isin(list(part2_new.index))]
        part4_0 = part4_0.T
        part4_0 = part4_0[part4_0.index.isin(list(part2_new.columns))]
        part4_0 = part4_0.T
        part4_0 = part4_0[closes.columns]
        part4_0 = drop_duplicates_index(part4_0)
        part4_0 = part4_0[sorted(list(part4_0.columns))]
        part4_0.to_parquet(homeplace.daily_data_file + "sts.parquet")
        logger.success("st更新完了")

        # 上市天数
        part5_close = pd.read_parquet(
            homeplace.update_data_file + "BasicFactor_Close.parquet"
        )
        part5_close = part5_close[part5_close.index < 20040101]
        part5_close.index = pd.to_datetime(part5_close.index, format="%Y%m%d")
        part5_close = pd.concat([part5_close, closes]).drop_duplicates()
        part5 = np.sign(part5_close).fillna(method="ffill").cumsum()
        part5 = part5[part5.index.isin(list(part2_new.index))]
        part5 = part5.T
        part5 = part5[part5.index.isin(list(part2_new.columns))]
        part5 = part5.T
        part5 = part5[closes.columns]
        part5 = drop_duplicates_index(part5)
        part5 = part5[sorted(list(part5.columns))]
        part5.to_parquet(homeplace.daily_data_file + "ages.parquet")
        logger.success("上市天数更新完了")


@retry
def download_single_day_style(day):
    """更新单日的数据"""
    try:
        style = pro.RMExposureDayGet(
            trade_date=str(day),
            fields="tradeDate,ticker,BETA,MOMENTUM,SIZE,EARNYILD,RESVOL,GROWTH,BTOP,LEVERAGE,LIQUIDTY,SIZENL",
        )
        time.sleep(1)
        return style
    except Exception:
        time.sleep(60)
        style = pro.RMExposureDayGet(
            trade_date=str(day),
            fields="tradeDate,ticker,BETA,MOMENTUM,SIZE,EARNYILD,RESVOL,GROWTH,BTOP,LEVERAGE,LIQUIDTY,SIZENL",
        )
        time.sleep(1)
        return style


def database_update_barra_files_dcube():
    fs = os.listdir(homeplace.barra_data_file)[0]
    fs = pd.read_parquet(homeplace.barra_data_file + fs)
    last_date = fs.index.max()
    last_date = datetime.datetime.strftime(last_date, "%Y%m%d")
    now = datetime.datetime.now()
    if now.hour < 17:
        now = now - pd.Timedelta(days=1)
    now = datetime.datetime.strftime(now, "%Y%m%d")
    logger.info(f"风格暴露数据上次更新到{last_date}，本次将更新到{now}")
    df0 = download_calendar(last_date, now)
    tradedates = sorted(list(set(df0.trade_date)))
    style_names = [
        "beta",
        "momentum",
        "size",
        "residualvolatility",
        "earningsyield",
        "growth",
        "booktoprice",
        "leverage",
        "liquidity",
        "nonlinearsize",
    ]
    ds = {k: [] for k in style_names}
    if len(tradedates) >= 1:
        for t in tqdm.auto.tqdm(tradedates):
            style = download_single_day_style(t)
            style.columns = style.columns.str.lower()
            style = style.rename(
                columns={
                    "earnyild": "earningsyield",
                    "tradedate": "date",
                    "ticker": "code",
                    "resvol": "residualvolatility",
                    "btop": "booktoprice",
                    "sizenl": "nonlinearsize",
                    "liquidty": "liquidity",
                }
            )
            style.date = pd.to_datetime(style.date, format="%Y%m%d")
            style.code = style.code.apply(add_suffix)
            sts = list(style.columns)[2:]
            for s in sts:
                ds[s].append(style.pivot(columns="code", index="date", values=s))
        for k, v in ds.items():
            old = pd.read_parquet(homeplace.barra_data_file + k + ".parquet")
            new = pd.concat(v)
            new = pd.concat([old, new])
            new.to_parquet(homeplace.barra_data_file + k + ".parquet")
        logger.success(f"风格暴露数据已经更新到{now}")
    else:
        logger.info("从上次更新到这次更新，还没有经过交易日。放假就好好休息吧，别跑代码了🤒")


def database_update_barra_files():
    fs = os.listdir(homeplace.barra_data_file)[0]
    fs = pd.read_parquet(homeplace.barra_data_file + fs)
    last_date = fs.index.max()
    last_date = datetime.datetime.strftime(last_date, "%Y%m%d")
    now = datetime.datetime.now()
    if now.hour < 17:
        now = now - pd.Timedelta(days=1)
    now = datetime.datetime.strftime(now, "%Y%m%d")
    logger.info(f"风格暴露数据上次更新到{last_date}，本次将更新到{now}")
    df0 = download_calendar(last_date, now)
    tradedates = sorted(list(set(df0.trade_date)))
    style_names = [
        "beta",
        "momentum",
        "size",
        "residualvolatility",
        "earningsyield",
        "growth",
        "booktoprice",
        "leverage",
        "liquidity",
        "nonlinearsize",
    ]
    ds = {k: [] for k in style_names}
    if len(tradedates) >= 1:
        codes = [convert_code(i)[0] for i in list(read_daily(open=1).columns)]
        style = rqdatac.get_factor_exposure(
            order_book_ids=codes,
            start_date=pd.Timestamp(last_date) + pd.Timedelta(days=1),
            end_date=now,
        ).reset_index()
        style = style.rename(
            columns={
                "earnings_yield": "earningsyield",
                "tradedate": "date",
                "ticker": "code",
                "residual_volatility": "residualvolatility",
                "book_to_price": "booktoprice",
                "non_linear_size": "nonlinearsize",
                "order_book_id": "code",
            }
        )
        style = style[style_names + ["date", "code"]]
        style.date = pd.to_datetime(style.date)
        style.code = style.code.apply(lambda x: convert_code(x)[0])
        for s in style_names:
            ds[s].append(style.pivot(columns="code", index="date", values=s))
        for k, v in ds.items():
            old = pd.read_parquet(homeplace.barra_data_file + k + ".parquet")
            new = pd.concat(v)
            new = pd.concat([old, new])
            new.to_parquet(homeplace.barra_data_file + k + ".parquet")
        logger.success(f"风格暴露数据已经更新到{now}")
    else:
        logger.info("从上次更新到这次更新，还没有经过交易日。放假就好好休息吧，别跑代码了🤒")


"""更新300、500、1000行情数据"""


@retry
def download_single_index(index_code: str):
    if index_code == "000300.SH":
        file = "沪深300"
    elif index_code == "000905.SH":
        file = "中证500"
    elif index_code == "000852.SH":
        file = "中证1000"
    else:
        file = index_code
    try:
        df = (
            pro.index_daily(ts_code=index_code)
            .sort_values("trade_date")
            .rename(columns={"trade_date": "date"})
        )
        df = df[["date", "close"]]
        df.date = pd.to_datetime(df.date, format="%Y%m%d")
        df = df.set_index("date")
        df = df.resample("M").last()
        df.columns = [file]
        return df
    except Exception:
        logger.warning("封ip了，请等待1分钟")
        time.sleep(60)
        df = (
            pro.index_daily(ts_code=index_code)
            .sort_values("trade_date")
            .rename(columns={"trade_date": "date"})
        )
        df = df[["date", "close"]]
        df.date = pd.to_datetime(df.date, format="%Y%m%d")
        df = df.set_index("date")
        df = df.resample("M").last()
        df.columns = [file]
        return df


def database_update_index_three():
    """读取三大指数的原始行情数据，返回并保存在本地"""
    hs300 = download_single_index("000300.SH")
    zz500 = download_single_index("000905.SH")
    zz1000 = download_single_index("000852.SH")
    res = pd.concat([hs300, zz500, zz1000], axis=1)
    new_date = datetime.datetime.strftime(res.index.max(), "%Y%m%d")
    res.to_parquet(homeplace.daily_data_file + "3510行情.parquet")
    logger.success(f"3510行情数据已经更新至{new_date}")


"""更新申万一级行业的行情"""


@retry
def download_single_industry_price(ind):
    try:
        df = pro.swindex_daily(code=ind)[["trade_date", "close"]].set_index(
            "trade_date"
        )
        df.columns = [ind]
        time.sleep(1)
        return df
    except Exception:
        time.sleep(60)
        df = pro.swindex_daily(code=ind)[["trade_date", "close"]].set_index(
            "trade_date"
        )
        df.columns = [ind]
        time.sleep(1)
        return df


def database_update_swindustry_prices():
    indus = []
    for ind in list(INDUS_DICT.keys()):
        df = download_single_industry_price(ind=ind)
        indus.append(df)
    indus = pd.concat(indus, axis=1).reset_index()
    indus.columns = ["date"] + list(indus.columns)[1:]
    indus = indus.set_index("date")
    indus = indus.dropna()
    indus.index = pd.to_datetime(indus.index, format="%Y%m%d")
    indus = indus.sort_index()
    indus.to_parquet(homeplace.daily_data_file + "申万各行业行情数据.parquet")
    new_date = datetime.datetime.strftime(indus.index.max(), "%Y%m%d")
    logger.success(f"申万一级行业的行情数据已经更新至{new_date}")


def database_update_zxindustry_prices():
    zxinds = ZXINDUS_DICT
    now = datetime.datetime.now()
    zxprices = []
    for k in list(zxinds.keys()):
        ind = rqdatac.get_price(
            k, start_date="2010-01-01", end_date=now, fields="close"
        )
        ind = (
            ind.rename(columns={"close": zxinds[k]})
            .reset_index(level=1)
            .reset_index(drop=True)
        )
        zxprices.append(ind)
    zxprice = reduce(lambda x, y: pd.merge(x, y, on=["date"], how="outer"), zxprices)
    zxprice.set_index(["date"]).to_parquet(
        homeplace.daily_data_file + "中信各行业行情数据.parquet"
    )
    new_date = datetime.datetime.strftime(zxprice.date.max(), "%Y%m%d")
    logger.success(f"中信一级行业的行情数据已经更新至{new_date}")


"""更新申万一级行业哑变量"""


@retry
def download_single_swindustry_member(ind):
    try:
        df = pro.index_member(index_code=ind)
        # time.sleep(1)
        return df
    except Exception:
        time.sleep(60)
        df = pro.index_member(index_code=ind)
        time.sleep(1)
        return df


def database_update_swindustry_member():
    dfs = []
    for ind in tqdm.auto.tqdm(INDUS_DICT.keys()):
        ff = download_single_swindustry_member(ind)
        ff = 生成每日分类表(ff, "con_code", "in_date", "out_date", "index_code")
        khere = ff.index_code.iloc[0]
        ff = ff.assign(khere=1)
        ff.columns = ["code", "index_code", "date", khere]
        ff = ff.drop(columns=["index_code"])
        dfs.append(ff)
    res = pd.merge(dfs[0], dfs[1], on=["code", "date"])
    dfs = pd.concat(dfs)
    trs = read_daily(tr=1, start=20040101)
    dfs = dfs[dfs.date.isin(trs.index)]
    dfs = dfs.sort_values(["date", "code"])
    dfs = dfs[["date", "code"] + list(dfs.columns)[2:]]
    dfs = dfs.fillna(0)
    dfs.reset_index(drop=True).to_parquet(
        homeplace.daily_data_file + "申万行业2021版哑变量.parquet"
    )
    new_date = dfs.date.max()
    new_date = datetime.datetime.strftime(new_date, "%Y%m%d")
    logger.success(f"申万一级行业成分股(哑变量)已经更新至{new_date}")


@retry
def download_single_index_member_monthly(code):
    file = homeplace.daily_data_file + INDEX_DICT[code] + "月成分股.parquet"
    old = pd.read_parquet(file)
    old_date = old.index.max()
    start_date = old_date + pd.Timedelta(days=1)
    end_date = datetime.datetime.now()
    if start_date >= end_date:
        logger.info(f"{INDEX_DICT[code]}月成分股无需更新，上次已经更新到了{start_date}")
    else:
        start_date, end_date = datetime.datetime.strftime(
            start_date, "%Y%m%d"
        ), datetime.datetime.strftime(end_date, "%Y%m%d")
        logger.info(f"{INDEX_DICT[code]}月成分股上次更新到{start_date},本次将更新到{end_date}")
        try:
            a = pro.index_weight(
                index_code=code, start_date=start_date, end_date=end_date
            )
            if a.shape[0] == 0:
                logger.info(f"{INDEX_DICT[code]}月成分股无需更新，上次已经更新到了{start_date}")
            else:
                time.sleep(1)
                a.trade_date = pd.to_datetime(a.trade_date, format="%Y%m%d")
                a = a.sort_values("trade_date").set_index("trade_date")
                a = (
                    a.groupby("con_code")
                    .resample("M")
                    .last()
                    .drop(columns=["con_code"])
                    .reset_index()
                )
                a = a.assign(num=1)
                a = (
                    a[["trade_date", "con_code", "num"]]
                    .rename(columns={"trade_date": "date", "con_code": "code"})
                    .pivot(columns="code", index="date", values="num")
                )
                a = pd.concat([old, a]).fillna(0)
                a.to_parquet(file)
                logger.success(f"已将{INDEX_DICT[code]}月成分股更新至{end_date}")
        except Exception:
            time.sleep(60)
            a = pro.index_weight(
                index_code=code, start_date=start_date, end_date=end_date
            )
            if a.shape[0] == 0:
                logger.info(f"{INDEX_DICT[code]}月成分股无需更新，上次已经更新到了{start_date}")
            else:
                time.sleep(1)
                a.trade_date = pd.to_datetime(a.trade_date, format="%Y%m%d")
                a = a.sort_values("trade_date").set_index("trade_date")
                a = (
                    a.groupby("con_code")
                    .resample("M")
                    .last()
                    .drop(columns=["con_code"])
                    .reset_index()
                )
                a = a.assign(num=1)
                a = (
                    a[["trade_date", "con_code", "num"]]
                    .rename(columns={"trade_date": "date", "con_code": "code"})
                    .pivot(columns="code", index="date", values="num")
                )
                a = pd.concat([old, a]).fillna(0)
                a.to_parquet(file)
                logger.success(f"已将{INDEX_DICT[code]}月成分股更新至{end_date}")


def database_update_index_members_monthly():
    for k in list(INDEX_DICT.keys()):
        download_single_index_member_monthly(k)


def download_single_index_member(code):
    file = homeplace.daily_data_file + INDEX_DICT[code] + "日成分股.parquet"
    if code.endswith(".SH"):
        code = code[:6] + ".XSHG"
    elif code.endswith(".SZ"):
        code = code[:6] + ".XSHE"
    now = datetime.datetime.now()
    df = rqdatac.index_components(
        code, start_date="20100101", end_date=now, market="cn"
    )
    ress = []
    for k, v in df.items():
        res = pd.DataFrame(1, index=[pd.Timestamp(k)], columns=v)
        ress.append(res)
    ress = pd.concat(ress)
    ress.columns = [convert_code(i)[0] for i in list(ress.columns)]
    tr = np.sign(read_daily(tr=1, start=20100101))
    tr = np.sign(tr + ress)
    now_str = datetime.datetime.strftime(now, "%Y%m%d")
    tr.to_parquet(file)
    logger.success(f"已将{INDEX_DICT[convert_code(code)[0]]}日成分股更新至{now_str}")


def database_update_index_members():
    for k in list(INDEX_DICT.keys()):
        download_single_index_member(k)


def database_save_final_factors(
    df: pd.DataFrame, name: str, order: int, freq: str = "月"
) -> None:
    """保存最终因子的因子值

    Parameters
    ----------
    df : pd.DataFrame
        最终因子值
    name : str
        因子的名字，如“适度冒险”
    order : int
        因子的序号
    """
    homeplace = HomePlace()
    path = (
        homeplace.final_factor_file
        + name
        + "_"
        + "多因子"
        + str(order)
        + "_"
        + freq
        + ".parquet"
    )
    df = df.drop_duplicates().dropna(how="all")
    df.to_parquet(path)
    final_date = df.index.max()
    final_date = datetime.datetime.strftime(final_date, "%Y%m%d")
    logger.success(f"今日计算的因子值保存，最新一天为{final_date}")


@retry
def download_single_day_money_flow(
    code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    try:
        df = pro.ashare_moneyflow(
            code=code,
            start_date=start_date,
            end_date=end_date,
            fields="trade_dt,buy_value_exlarge_order,sell_value_exlarge_order,buy_value_large_order,sell_value_large_order,buy_value_med_order,sell_value_med_order,buy_value_small_order,sell_value_small_order",
        ).iloc[::-1, :]
        if df.shape[0] > 0:
            df = df.assign(code=code)
            return df
        time.sleep(0.1)
    except Exception:
        time.sleep(60)
        df = pro.ashare_moneyflow(
            code=code,
            start_date=start_date,
            end_date=end_date,
            fields="trade_dt,buy_value_exlarge_order,sell_value_exlarge_order,buy_value_large_order,sell_value_large_order,buy_value_med_order,sell_value_med_order,buy_value_small_order,sell_value_small_order",
        ).iloc[::-1, :]
        if df.shape[0] > 0:
            df = df.assign(code=code)
            return df
        time.sleep(0.1)


def database_update_money_flow():
    trs = read_daily(tr=1)
    codes = sorted(list(trs.columns))
    codes = [i for i in codes if i[0] != "8"]
    old = read_money_flow(buy=1, small=1)
    old_enddate = old.index.max()
    old_enddate_str = datetime.datetime.strftime(old_enddate, "%Y%m%d")
    new_trs = trs[trs.index > old_enddate]
    start_date = new_trs.index.min()
    start_date_str = datetime.datetime.strftime(start_date, "%Y%m%d")
    now = datetime.datetime.now()
    now_str = datetime.datetime.strftime(now, "%Y%m%d")
    logger.info(f"上次资金流数据更新到{old_enddate_str}，本次将从{start_date_str}更新到{now_str}")
    dfs = []
    for code in tqdm.auto.tqdm(codes):
        df = download_single_day_money_flow(
            code=code, start_date=start_date_str, end_date=now_str
        )
        dfs.append(df)
    dfs = [i for i in dfs if i is not None]
    if len(dfs) > 0:
        dfs = pd.concat(dfs)
        dfs = dfs.rename(columns={"trade_dt": "date"})
        dfs.date = pd.to_datetime(dfs.date, format="%Y%m%d")
        ws = [i for i in list(dfs.columns) if i not in ["date", "code"]]
        for w in ws:
            old = pd.read_parquet(homeplace.daily_data_file + w[:-6] + ".parquet")
            new = dfs.pivot(index="date", columns="code", values=w)
            new = pd.concat([old, new])
            new = new[sorted(list(new.columns))]
            new.to_parquet(homeplace.daily_data_file + w[:-6] + ".parquet")
        logger.success(f"已经将资金流数据更新到{now_str}")
    else:
        logger.warning(f"从{start_date_str}到{now_str}暂无数据")


def database_update_zxindustry_member():
    """更新中信一级行业的成分股"""
    old_codes = pd.read_parquet(homeplace.daily_data_file + "中信一级行业哑变量代码版.parquet")
    old_names = pd.read_parquet(homeplace.daily_data_file + "中信一级行业哑变量名称版.parquet")
    old_enddate = old_codes.date.max()
    old_enddate_str = datetime.datetime.strftime(old_enddate, "%Y%m%d")
    now = datetime.datetime.now()
    now_str = datetime.datetime.strftime(now, "%Y%m%d")
    logger.info(f"中信一级行业数据，上次更新到了{old_enddate_str}，本次将更新至{now_str}")
    start_date = old_enddate + pd.Timedelta(days=1)
    start_date = datetime.datetime.strftime(start_date, "%Y%m%d")
    codes = list(
        set(rqdatac.all_instruments(type="CS", market="cn", date=None).order_book_id)
    )
    trs = read_daily(tr=1)
    trs = trs[trs.index > old_enddate]
    dates = list(trs.index)
    dfs_codes = []
    dfs_names = []
    if len(dates) >= 1:
        for date in tqdm.auto.tqdm(dates):
            df = rqdatac.get_instrument_industry(
                codes, source="citics_2019", date=date, level=1
            )
            if df.shape[0] > 0:
                df_code = df.first_industry_code.to_frame(date)
                df_name = df.first_industry_name.to_frame(date)
                dfs_codes.append(df_code)
                dfs_names.append(df_name)
        dfs_codes = pd.concat(dfs_codes, axis=1)
        dfs_names = pd.concat(dfs_names, axis=1)

        def new_get_dummies(df):
            dums = []
            for col in tqdm.auto.tqdm(list(df.columns)):
                series = df[col]
                dum = pd.get_dummies(series)
                dum = dum.reset_index()
                dum = dum.assign(date=col)
                dums.append(dum)
            dums = pd.concat(dums)
            return dums

        dfs_codes = new_get_dummies(dfs_codes)
        dfs_names = new_get_dummies(dfs_names)

        a = read_daily(tr=1, start=20100101)

        def save(df, old, file):
            df = df.rename(columns={"order_book_id": "code"})
            df = df[["date", "code"] + sorted(list(df.columns)[1:-1])]
            df.code = df.code.apply(lambda x: convert_code(x)[0])
            df = pd.concat([old, df], ignore_index=True)
            df = df[df.date.isin(list(a.index))]
            df=df.reset_index(drop=True).replace(True,1).replace(False,0)
            df.to_parquet(homeplace.daily_data_file + file)
            return df

        dfs_codes = save(dfs_codes, old_codes, "中信一级行业哑变量代码版.parquet")
        dfs_names = save(dfs_names, old_names, "中信一级行业哑变量名称版.parquet")
        logger.success(f"中信一级行业数据已经更新至{now_str}了")
    else:
        logger.warning(f"从{start_date}到{now_str}暂无数据")


def database_update_idiosyncratic_ret():
    pb = read_daily(pb=1, start=20100101)
    cap = read_daily(flow_cap=1, start=20100101).dropna(how="all")
    fama = pure_fama([cap, pb])
    fama().to_parquet(homeplace.daily_data_file + "idiosyncratic_ret.parquet")
    logger.success("特质收益率已经更新完成")


def database_update_illiquidity():
    ret = read_daily(ret=1, start=20100101)
    money = read_daily(money=1, start=20100101)
    illi = ret.abs() / money
    illi.to_parquet(homeplace.daily_data_file + "illiquidity.parquet")
    logger.success("非流动性数据已经更新完成")


def database_update_industry_rets_for_stock():
    closes = read_daily(close=1, start=20100101)
    indu_rets = read_zxindustry_prices(monthly=0, start=20100101).pct_change()
    indus = {
        k: pd.DataFrame(
            {code: list(indu_rets[k]) for code in list(closes.columns)},
            index=indu_rets.index,
        )
        for k in indu_rets.keys()
    }
    a = get_industry_dummies(daily=1, zxindustry=1, start=20100101)
    res = {k: indus[k] * a[k] for k in indus.keys()}
    res = reduce(
        lambda x, y: select_max(x.fillna(-1000), y.fillna(-1000)), list(res.values())
    ).replace(-1000, np.nan)
    res.to_parquet(homeplace.daily_data_file + "股票对应中信一级行业每日收益率.parquet")
    logger.success("股票对应中信一级行业每日收益率已经更新完")
    indu_rets = read_swindustry_prices(monthly=0, start=20100101).pct_change()
    indus = {
        k: pd.DataFrame(
            {code: list(indu_rets[k]) for code in list(closes.columns)},
            index=indu_rets.index,
        )
        for k in indu_rets.keys()
    }
    a = get_industry_dummies(daily=1, swindustry=1, start=20100101)
    res = {k: indus[k] * a[k] for k in indus.keys()}
    res = reduce(
        lambda x, y: select_max(x.fillna(-1000), y.fillna(-1000)), list(res.values())
    ).replace(-1000, np.nan)
    res.to_parquet(homeplace.daily_data_file + "股票对应申万一级行业每日收益率.parquet")
    logger.success("股票对应申万一级行业每日收益率已经更新完")


class FactorReader:
    def __init__(
        self,
        user: str = "admin",
        password: str = "quest",
        host: str = "127.0.0.1",
        port: str = "8812",
        database: str = "qdb",
    ) -> None:
        """通过postgre的psycopg2驱动连接questdb数据库

        Parameters
        ----------
        user : str, optional
            用户名, by default "admin"
        password : str, optional
            密码, by default "quest"
        host : str, optional
            地址, by default "43.143.223.158"
        port : str, optional
            端口, by default "8812"
        database : str, optional
            数据库, by default "qdb"
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def __connect(self):
        conn = pg.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        return conn

    def update_factor(self, table_name: str, df: pd.DataFrame):
        tables = self.__get_data("show tables").table.tolist()
        if table_name in tables:
            logger.info(f"{table_name}已经存在了，即将更新")
            old_end = self.__get_data(f"select max(date) from {table_name}").iloc[0, 0]
            new = df[df.index > old_end]
            new = new.stack().reset_index()
            new.columns = ["date", "code", "fac"]
        else:
            logger.info(f"{table_name}第一次上传")
            new = df.stack().reset_index()
            new.columns = ["date", "code", "fac"]
        self.__write_via_df(new, table_name)

    def __write_via_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        symbols=None,
        tuple_col=None,
    ) -> None:
        """通过questdb的python库直接将dataframe写入quested数据库

        Parameters
        ----------
        df : pd.DataFrame
            要写入的dataframe
        table_name : str
            questdb中该表的表名
        symbols : Union[str, bool, List[int], List[str]], optional
            为symbols的那些列的名称, by default None
        tuple_col : Union[str, List[str]], optional
            数据类型为tuple或list的列的名字, by default None
        """
        if tuple_col is None:
            ...
        elif isinstance(tuple_col, str):
            df[tuple_col] = df[tuple_col].apply(str)
        else:
            for t in tuple_col:
                df[t] = df[t].apply(str)
        if symbols is not None:
            with qdbing.Sender(self.host, 9009) as sender:
                sender.dataframe(df, table_name=table_name, symbols=symbols)
        else:
            with qdbing.Sender(self.host, 9009) as sender:
                sender.dataframe(df, table_name=table_name)

    @retry(stop=stop_after_attempt(10))
    def __get_data(self, sql_order: str) -> pd.DataFrame:
        """以sql命令的方式，从数据库中读取数据

        Parameters
        ----------
        sql_order : str
            sql命令

        Returns
        -------
        pd.DataFrame
            读取的结果
        """
        conn = self.__connect()
        cursor = conn.cursor()
        cursor.execute(sql_order)
        df_data = cursor.fetchall()
        columns = [i[0] for i in cursor.description]
        df = pd.DataFrame(df_data, columns=columns)
        return df

    def add_token(self, tokens: List[str], users: List[str]):
        tus = pd.DataFrame({"token": tokens, "user": users})
        self.__write_via_df(tus, "tokenlines")


def database_update_index_weight():
    opens = read_daily(open=1).resample("M").last()
    dates = [datetime.datetime.strftime(i, "%Y%m%d") for i in list(opens.index)]
    df1s = []
    df2s = []
    df3s = []
    for i in tqdm.auto.tqdm(dates):
        df1s.append(
            rqdatac.index_weights(convert_code("000300.SH")[0], date=i).to_frame(i)
        )
        df2s.append(
            rqdatac.index_weights(convert_code("000905.SH")[0], date=i).to_frame(i)
        )
        try:
            df3s.append(
                rqdatac.index_weights(convert_code("000852.SH")[0], date=i).to_frame(i)
            )
        except Exception:
            ...

    def deal(df):
        df = pd.concat(df, axis=1).T
        df.index = pd.to_datetime(df.index)
        df.columns = [convert_code(i)[0] for i in df.columns]
        return df

    df1, df2, df3 = list(map(deal, [df1s, df2s, df3s]))
    df1.to_parquet(homeplace.daily_data_file + "沪深300成分股权重.parquet")
    df2.to_parquet(homeplace.daily_data_file + "中证500成分股权重.parquet")
    df3.to_parquet(homeplace.daily_data_file + "中证1000成分股权重.parquet")
    logger.success("指数成分股权重更新完了")
