## 更新日志🗓 — v4


* v4.0.1 — 2023.7.3

> 1. 修正了convert_tick_by_tick_data_to_parquet中成交额与成交量的混淆错误
> 2. 修复了database_update_zxindustry_member中的一些异常
> 3. 修复了pure_fall_nature中的一些异常
> 4. 取消了对matplotlib库的依赖


* v4.0.0 — 2023.6.28

> 1. 修复了初始化时可能产生的报错，将初始化函数更名为ini，可通过如下语句初始化
>
>    ```python
>    import pure_ocean_breeze as p
>       
>    p.ini()
>    ```
> 2. 初始化函数与`Homeplace`参数新增了存储逐笔数据的路径
> 3. 增加了使用逐笔数据计算因子值的代码框架`pure_fall_nature`，并可以使用逐笔数据合成任意秒线、分钟线、小时线
> 4. write_data部分新增了`convert_tick_by_tick_data_to_parquet`、`convert_tick_by_tick_data_daily`、`convert_tick_by_tick_data_monthly`三个更新逐笔数据的函数
> 5. 修复了database_update_daily_files中关于更新pe_ttm数据的错误
> 6. 修复了pure_moon中截止日期非最新日期时，关于多头超均收益的错误
> 7. 修复了pure_fall_frequent中确实日期仅为异常日期时的bug
> 8. ClickHouseClient中新增了秒级数据的show_all_dates
> 9. 将对pandas的依赖限制在了1.5.3及以下版本