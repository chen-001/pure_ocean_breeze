__updated__ = "2022-09-13 16:43:45"

try:
    import rqdatac

    rqdatac.init()
except Exception:
    print("暂时未连接米筐")
from loguru import logger
import os
import time
import datetime
import numpy as np
import pandas as pd
import scipy.io as scio
from sqlalchemy import FLOAT, INT, VARCHAR, BIGINT
from tenacity import retry
import pickledb
import tqdm
from functools import reduce
import dcube as dc
from pure_ocean_breeze.legacy_version.v3p1.state.homeplace import HomePlace

homeplace = HomePlace()
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
from pure_ocean_breeze.legacy_version.v3p1.data.read_data import read_daily, read_money_flow
from pure_ocean_breeze.legacy_version.v3p1.data.dicts import INDUS_DICT, INDEX_DICT, ZXINDUS_DICT
from pure_ocean_breeze.legacy_version.v3p1.data.tools import 生成每日分类表, add_suffix, convert_code


def database_update_minute_data_to_clickhouse_and_questdb(kind: str) -> None:
    """使用米筐更新分钟数据至clickhouse和questdb中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'

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
        time_slice=None,
    )
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
    qdb = Questdb()
    ts.date = ts.date.astype(int).astype(str)
    ts.num = ts.num.astype(int).astype(str)
    qdb.write_via_csv(ts, f"minute_data_{kind}")
    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")


def database_update_minute_data_to_postgresql(kind: str) -> None:
    """使用米筐更新分钟数据至postgresql中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'

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
    qdb = Questdb()
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
        time_slice=None,
    )
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
    ts.code = ts.code.str.replace(".XSHE", ".SZ")
    ts.code = ts.code.str.replace(".XSHG", ".SH")
    # 数据写入数据库
    pgdb = PostgreSQL("minute_data")
    ts.to_sql(
        f"minute_data_{kind}",
        pgdb.engine,
        if_exists="append",
        index=False,
        dtype={
            "code": VARCHAR(9),
            "date": p.INT,
            "open": FLOAT,
            "high": FLOAT,
            "low": FLOAT,
            "close": FLOAT,
            "amount": FLOAT,
            "money": FLOAT,
            "num": INT,
        },
    )
    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")


def database_update_minute_data_to_questdb(kind: str) -> None:
    """使用米筐更新分钟数据至questdb中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'

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
    qdb = Questdb()
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
        time_slice=None,
    )
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
    qdb = Questdb()
    qdb.write_via_csv(df, f"minute_data_{kind}")
    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")


def database_update_minute_data_to_mysql(kind: str) -> None:
    """使用米筐更新分钟数据至mmysql中

    Parameters
    ----------
    kind : str
        更新股票分钟数据或指数分钟数据，股票则'stock'，指数则'index'

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
    # 连接2个数据库
    sqlsa = sqlConfig("minute_data_stock_alter")
    sqlia = sqlConfig("minute_data_index_alter")
    last_date = max(sqlsa.show_tables(full=False))
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
        time_slice=None,
    )
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
    ts.code = ts.code.str.replace(".XSHE", ".SZ")
    ts.code = ts.code.str.replace(".XSHG", ".SH")
    codes = list(set(ts.code))
    dates = list(set(ts.date))
    # 数据写入数据库
    fails = []
    # 股票
    if kind == "stock":
        # 把每天写入每天所有股票一张表
        for date in dates:
            dfi = ts[ts.date == date]
            try:
                dfi.drop(columns=["date"]).to_sql(
                    name=str(date),
                    con=sqlsa.engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        "code": VARCHAR(9),
                        "open": FLOAT,
                        "high": FLOAT,
                        "low": FLOAT,
                        "close": FLOAT,
                        "amount": FLOAT,
                        "money": FLOAT,
                        "num": INT,
                    },
                )
            except Exception:
                try:
                    if sqlsa.get_data(date).shape[0] == 0:
                        dfi.drop(columns=["date"]).to_sql(
                            name=str(date),
                            con=sqlsa.engine,
                            if_exists="replace",
                            index=False,
                            dtype={
                                "code": VARCHAR(9),
                                "open": FLOAT,
                                "high": FLOAT,
                                "low": FLOAT,
                                "close": FLOAT,
                                "amount": FLOAT,
                                "money": FLOAT,
                                "num": INT,
                            },
                        )
                except Exception:
                    fails.append(date)
                    logger.warning(f"股票{date}写入失败了，请检查")
    # 指数
    else:
        # 把每天写入每天所有指数一张表
        for date in dates:
            dfi = ts[ts.date == date]
            try:
                dfi.drop(columns=["date"]).to_sql(
                    name=str(date),
                    con=sqlia.engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        "code": VARCHAR(9),
                        "open": FLOAT,
                        "high": FLOAT,
                        "low": FLOAT,
                        "close": FLOAT,
                        "amount": FLOAT,
                        "money": FLOAT,
                        "num": INT,
                    },
                )
            except Exception:
                try:
                    if sqlia.get_data(date).shape[0] == 0:
                        dfi.drop(columns=["date"]).to_sql(
                            name=str(date),
                            con=sqlia.engine,
                            if_exists="replace",
                            index=False,
                            dtype={
                                "code": VARCHAR(9),
                                "open": FLOAT,
                                "high": FLOAT,
                                "low": FLOAT,
                                "close": FLOAT,
                                "amount": FLOAT,
                                "money": FLOAT,
                                "num": INT,
                            },
                        )
                except Exception:
                    fails.append(date)
                    logger.warning(f"指数{date}写入失败了，请检查")

    # 获取剩余使用额
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    user12 = round(user2 - user1, 2)
    logger.info(f"今日已使用rqsdk流量{user2}MB，本项更新消耗流量{user12}MB")


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
            ],
        )
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2 = pro.daily_basic(
            trade_date=day,
            fields=["ts_code", "trade_date", "turnover_rate_f", "float_share"],
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
            ],
        )
        # 换手率，流通股本，换手率要除以100，流通股本要乘以10000
        df2 = pro.daily_basic(
            trade_date=day,
            fields=["ts_code", "trade_date", "turnover_rate_f", "float_share"],
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


def database_update_daily_files(startdate: str = None, enddate: str = None) -> None:
    """更新数据库中的日频数据

    Parameters
    ----------
    startdate : str, optional
        形如'20220501'，不填写将自动识别, by default None
    enddate : str, optional
        形如'20220701'，不填写将自动识别, by default None

    Raises
    ------
    `ValueError`
        如果上次更新到本次更新没有新的交易日，将报错
    """
    read_daily.clear_cache()
    homeplace = HomePlace()
    config = pickledb.load(homeplace.update_data_file + "database_config.db", False)
    if startdate:
        ...
    else:
        startdate = config.get("daily_enddate")
        logger.info(
            f"上次更新到{datetime.datetime.strftime(pd.Timestamp(startdate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
        )
    if enddate:
        ...
    else:
        enddate = datetime.datetime.now()
        if enddate.hour < 17:
            enddate = enddate - pd.Timedelta(days=1)
        else:
            ...
        enddate = datetime.datetime.strftime(enddate, "%Y%m%d")
        logger.info(
            f"本次将更新到{datetime.datetime.strftime(pd.Timestamp(enddate),format='%Y-%m-%d')}"
        )
    # 交易日历
    df0 = download_calendar(startdate, enddate)
    tradedates = sorted(list(set(df0.trade_date)))
    if len(tradedates) > 1:
        # 存储每天数据
        df1s = []
        df2s = []
        for day in tqdm.tqdm(tradedates, desc="正在下载日频数据"):
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
        raise ValueError("从上次更新到这次更新，还没有经过交易日。放假就好好休息吧，别跑代码了🤒")
    df1s.tradestatus = (df1s.tradestatus == "交易") + 0
    df2s = df2s.rename(columns={"ts_code": "code"})
    df1s.trade_date = df1s.trade_date.apply(int)
    df2s.trade_date = df2s.trade_date.apply(int)
    both_codes = list(set(df1s.code) & set(df2s.code))
    df1s = df1s[df1s.code.isin(both_codes)]
    df2s = df2s[df2s.code.isin(both_codes)]
    # st股
    df3 = pro.ashare_st()

    suffix_path = homeplace.update_data_file
    suffix_path_save = homeplace.daily_data_file
    codes = list(scio.loadmat(suffix_path_save + "AllStockCode.mat").values())[3]
    codes = [i[0] for i in codes[0]]
    days = list(scio.loadmat(suffix_path_save + "TradingDate_Daily.mat").values())[3]
    days = [i[0] for i in days]

    def read_mat(path):
        col = list(scio.loadmat(suffix_path_save + "AllStockCode.mat").values())[3]
        index = list(scio.loadmat(suffix_path_save + "TradingDate_Daily.mat").values())[
            3
        ]
        col = [i[0] for i in col[0]]
        if len(index) > 1000:
            index = [i[0] for i in index]
        else:
            index = index[0]
        path = suffix_path_save + path
        data = list(scio.loadmat(path).values())[3]
        data = pd.DataFrame(data, index=index, columns=col)
        return data

    def to_mat(df, row, filename=None, ind="trade_date", col="code"):
        df = df[[ind, col, row]].set_index([ind, col])
        df = df.unstack()
        df.columns = [i[1] for i in list(df.columns)]
        old = read_mat(filename)
        new = pd.concat([old, df])
        scio.savemat(
            suffix_path_save + filename, {"data": new.to_numpy()}, do_compression=True
        )
        logger.success(filename + "已更新")
        return new

    # 股票日行情（未复权高开低收，复权高开低收，交易状态，成交量）
    part1 = df1s.copy()
    # 未复权开盘价
    opens = to_mat(part1, "open", "AllStock_DailyOpen.mat")
    # 未复权最高价
    highs = to_mat(part1, "high", "AllStock_DailyHigh.mat")
    # 未复权最低价
    lows = to_mat(part1, "low", "AllStock_DailyLow.mat")
    # 未复权收盘价
    closes = to_mat(part1, "close", "AllStock_DailyClose.mat")
    # 成交量
    volumes = to_mat(part1, "volume", "AllStock_DailyVolume.mat")
    # 复权开盘价
    diopens = to_mat(part1, "adjopen", "AllStock_DailyOpen_dividend.mat")
    # 复权最高价
    dihighs = to_mat(part1, "adjhigh", "AllStock_DailyHigh_dividend.mat")
    # 复权最低价
    dilows = to_mat(part1, "adjlow", "AllStock_DailyLow_dividend.mat")
    # 复权收盘价
    dicloses = to_mat(part1, "adjclose", "AllStock_DailyClose_dividend.mat")
    # 交易状态
    status = to_mat(part1, "tradestatus", "AllStock_DailyStatus.mat")

    # 换手率
    part2 = df2s[["trade_date", "code", "turnover_rate_f"]]
    part2 = part2.set_index(["trade_date", "code"]).unstack()
    part2.columns = [i[1] for i in list(part2.columns)]
    part2 = part2 / 100
    part2_old = read_mat("AllStock_DailyTR.mat")
    part2_new = pd.concat([part2_old, part2])
    # part2_new=part2_new.dropna(how='all',axis=1)
    part2_new = part2_new[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyTR.mat",
        {"data": part2_new.to_numpy()},
        do_compression=True,
    )
    logger.success("换手率更新完成")

    # #交易日历和股票代码
    # part2_new=part2_new.reset_index()
    # part2_new.columns=['date']+list(part2_new.columns)[1:]
    # part2_new.to_feather('日历与代码暂存.feather')

    # 流通股数
    # 读取新的流通股变动数
    part3 = df2s[["trade_date", "code", "float_share"]]
    part3 = part3.set_index(["trade_date", "code"]).unstack()
    part3.columns = [i[1] for i in list(part3.columns)]
    part3 = part3 * 10000
    part3_old = read_mat("AllStock_DailyAShareNum.mat")
    part3_new = pd.concat([part3_old, part3])
    # part2_new=part2_new.dropna(how='all',axis=1)
    part3_new = part3_new[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyAShareNum.mat",
        {"data": part3_new.to_numpy()},
        do_compression=True,
    )
    logger.success("流通股数更新完成")

    # st
    part4 = df3[["s_info_windcode", "entry_dt", "remove_dt"]]
    part4 = part4.sort_values("s_info_windcode")
    part4.remove_dt = part4.remove_dt.fillna(enddate).astype(int)
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

    tqdm.tqdm.pandas()
    part4 = part4.groupby(["code", "he"]).progress_apply(single)
    part4.date = part4.date.dt.strftime("%Y%m%d").astype(int)
    part4 = part4[part4.date.isin(list(part2_new.index))]
    part4 = part4.reset_index(drop=True)
    part4 = part4.assign(st=1)

    part4 = part4.drop_duplicates(subset=["date", "code"]).pivot(
        index="date", columns="code", values="st"
    )

    part4_0 = pd.DataFrame(0, columns=part2_new.columns, index=part2_new.index)
    part4_0 = part4_0 + part4
    # old=read_mat('AllStock_DailyST.mat')
    # part4_new=pd.concat([old,part4_0])
    part4_0 = part4_0.replace(np.nan, 0)
    part4_0 = part4_0[part4_0.index.isin(list(part2_new.index))]
    part4_0 = part4_0.T
    part4_0 = part4_0[part4_0.index.isin(list(part2_new.columns))]
    part4_0 = part4_0.T
    # part4_0=part4_0.dropna(how='all',axis=1)
    part4_0 = part4_0[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyST.mat",
        {"data": part4_0.to_numpy()},
        do_compression=True,
    )
    logger.success("st更新完了")

    # 上市天数
    part5_close = pd.read_feather(suffix_path + "BasicFactor_Close.txt").set_index(
        "index"
    )
    part5_close = part5_close[part5_close.index < 20040101]
    part5_close = pd.concat([part5_close, closes])
    # part5_close.reset_index().to_feather(suffix_path+'BasicFactor_Close.txt')
    part5 = np.sign(part5_close).fillna(method="ffill").cumsum()
    part5 = part5[part5.index.isin(list(part2_new.index))]
    part5 = part5.T
    part5 = part5[part5.index.isin(list(part2_new.columns))]
    part5 = part5.T
    # part5=part5.dropna(how='all',axis=1)
    part5 = part5[closes.columns]
    scio.savemat(
        suffix_path_save + "AllStock_DailyListedDate.mat",
        {"data": part5.to_numpy()},
        do_compression=True,
    )
    logger.success("上市天数更新完了")

    # 交易日历和股票代码
    scio.savemat(
        suffix_path_save + "TradingDate_Daily.mat",
        {"data": part2_new.index.to_numpy()},
        do_compression=True,
    )
    scio.savemat(
        suffix_path_save + "AllStockCode.mat",
        {"data": part2_new.columns.to_numpy()},
        do_compression=True,
    )
    enddate = pd.Timestamp(enddate) + pd.Timedelta(days=1)
    enddate = datetime.datetime.strftime(enddate, "%Y%m%d")
    config.set("daily_enddate", enddate)
    config.dump()
    logger.success("交易日历和股票代码更新完了")
    read_daily.clear_cache()
    logger.success(
        f"日频数据已更新，现在最新的是{datetime.datetime.strftime(pd.Timestamp(enddate)-pd.Timedelta(days=1),format='%Y-%m-%d')}"
    )


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


def database_update_barra_files():
    fs = os.listdir(homeplace.barra_data_file)[0]
    fs = pd.read_feather(homeplace.barra_data_file + fs)
    fs.columns = ["date"] + list(fs.columns)[1:]
    fs = fs.set_index("date")
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
    for t in tqdm.tqdm(tradedates):
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
        old = pd.read_feather(homeplace.barra_data_file + k + ".feather")
        old.columns = ["date"] + list(old.columns)[1:]
        old = old.set_index("date")
        new = pd.concat(v)
        new = pd.concat([old, new])
        new.reset_index().to_feather(homeplace.barra_data_file + k + ".feather")
    logger.success(f"风格暴露数据已经更新到{now}")


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
    res.reset_index().to_feather(homeplace.daily_data_file + "3510行情.feather")
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
    indus.reset_index().to_feather(homeplace.daily_data_file + "申万各行业行情数据.feather")
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
    zxprice.reset_index(drop=True).to_feather(
        homeplace.daily_data_file + "中信各行业行情数据.feather"
    )
    new_date = datetime.datetime.strftime(zxprice.date.max(), "%Y%m%d")
    logger.success(f"中信一级行业的行情数据已经更新至{new_date}")


"""更新申万一级行业哑变量"""


@retry
def download_single_industry_member(ind):
    try:
        df = pro.index_member(index_code=ind)
        # time.sleep(1)
        return df
    except Exception:
        time.sleep(60)
        df = pro.index_member(index_code=ind)
        time.sleep(1)
        return df


def database_update_industry_member():
    dfs = []
    for ind in tqdm.tqdm(INDUS_DICT.keys()):
        ff = download_single_industry_member(ind)
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
    dfs.reset_index(drop=True).to_feather(
        homeplace.daily_data_file + "申万行业2021版哑变量.feather"
    )
    new_date = dfs.date.max()
    new_date = datetime.datetime.strftime(new_date, "%Y%m%d")
    logger.success(f"申万一级行业成分股(哑变量)已经更新至{new_date}")


@retry
def download_single_index_member_monthly(code):
    file = homeplace.daily_data_file + INDEX_DICT[code] + "月成分股.feather"
    old = pd.read_feather(file).set_index("index")
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
                a.reset_index().to_feather(file)
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
                a.reset_index().to_feather(file)
                logger.success(f"已将{INDEX_DICT[code]}月成分股更新至{end_date}")


def database_update_index_members_monthly():
    for k in list(INDEX_DICT.keys()):
        download_single_index_member_monthly(k)


def download_single_index_member(code):
    file = homeplace.daily_data_file + INDEX_DICT[code] + "日成分股.feather"
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
    tr.reset_index().to_feather(file)
    logger.success(f"已将{INDEX_DICT[convert_code(code)[0]]}日成分股更新至{now_str}")


def database_update_index_members():
    for k in list(INDEX_DICT.keys()):
        download_single_index_member(k)


def database_save_final_factors(df: pd.DataFrame, name: str, order: int) -> None:
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
    path = homeplace.final_factor_file + name + "_" + "多因子" + str(order) + ".feather"
    df.reset_index().to_feather(path)
    final_date = df.index.max()
    final_date = datetime.datetime.strftime(final_date, "%Y%m%d")
    config = pickledb.load(homeplace.update_data_file + "database_config.db", False)
    config.set("data_refresh", "done")
    config.dump()
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
    for code in tqdm.tqdm(codes):
        df = download_single_day_money_flow(
            code=code, start_date=start_date_str, end_date=now_str
        )
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs = dfs.rename(columns={"trade_dt": "date"})
    dfs.date = pd.to_datetime(dfs.date, format="%Y%m%d")
    ws = [i for i in list(dfs.columns) if i not in ["date", "code"]]
    for w in ws:
        old = pd.read_feather(
            homeplace.daily_data_file + w[:-6] + ".feather"
        ).set_index("date")
        new = dfs.pivot(index="date", columns="code", values=w)
        new = pd.concat([old, new])
        new = new[sorted(list(new.columns))]
        new.reset_index().to_feather(homeplace.daily_data_file + w[:-6] + ".feather")
    logger.success(f"已经将资金流数据更新到{now_str}")
