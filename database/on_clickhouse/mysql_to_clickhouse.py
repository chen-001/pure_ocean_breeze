from pure_ocean_breeze.pure_ocean_breeze import *

sql=sqlConfig('minute_data')
chc=ClickHouseClient('minute_data')
codes=sql.show_tables(full=False)

fails=[]
for code in tqdm.tqdm(codes):
    try:
        df=sql.get_data(code)
        (np.around(df,2)*100).ffill().dropna().astype(int).assign(code=code).to_sql('minute_data',chc.engine,if_exists='append',index=False)
    except Exception:
        fails.append(code)
        logger.warning(f'{code}失败了，请检查')
logger.success('存储完啦')

