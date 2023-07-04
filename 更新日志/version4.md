## 更新日志🗓 — v4


* v4.0.4 — 2023.7.4

> 1. 优化了show_corrs_with_old的结果展示排序
> 2. 修复了pure_fall_nature中关于money与amount的错误
> 3. 给pure_fall_nature的get_daily_factors方法增加了fields参数，用于指定要读取的字段，以节约内存（使用duckdb实现）
> 4. 将pure_fall_nature中的并行方法改为使用concurrent


* v4.0.3 — 2023.7.4

> 1. 新增了全新的因子成果数据库FactorDone，每个最终复合因子，都附带其细分因子
> 2. 对show_corr函数的默认计算方式进行了提速，并新增old_way参数，用于使用旧版方式计算
> 3. 将show_corr、show_corrs、show_corrs_with_old函数中相关系数的默认计算方式调整为pearson相关系数
> 4. 对show_corrs_with_old的内容进行了升级，以支持计算因子与已有因子的细分因子之间的相关系数


* v4.0.2 — 2023.7.3

> 1. 暂时取消了对numpy的强制依赖


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