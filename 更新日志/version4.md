## 更新日志🗓 — v4


* v4.0.8 — 2023.10.14

> 1. 修复了Questdb读取到None时不能识别的bug
> 2. 修正了read_daily计算振幅的公式错误
> 3. 修复了read_market中读取开盘价时的错误
> 4. 升级了pure_moon因子回测结果展示，删去了因子截面标准差，新增了各组月均超额收益项
> 5. 给follow_tests新增了groups_num参数，用于指定分组数量；新增了without_industry参数，指定不在行业成分股内做测试


* v4.0.7 — 2023.7.17

> 1. 给pure_fall_frequent和pure_fall_nature新增了use_mpire参数，可以使用mpire库开启并行
> 2. 修复了pure_fall_frequent的select_one_calculate方法中，可能存在的返回类型为字符串的问题
> 3. 修复了pure_fall_nature中全新因子运算完拼接时的bug


* v4.0.6 — 2023.7.14

> 1. 修复了pure_dawn读取已有因子值时`__call__`方法返回错误的问题
> 2. 修复了pure_fall_frequent中关于计算因子值的bug
> 3. 修复了optimize_many_days函数和pure_linprog的run方法不受STATES['START']影响的bug
> 4. 随着时间的推移，将STATES['START']值修改为20140101
> 5. 更新了依赖库


* v4.0.5 — 2023.7.7

> 1. 修复了pure_fall_nature中缺少fields参数和参数传递的bug
> 2. 修复了pure_fall_nature中拼接新旧因子时的错误
> 3. 删去了pure_fall_nature中get_daily_factors的系统通知
> 4. 将pure_fall_frequent中的并行改为使用concurrent
> 5. 对pure_fall_frequent中的get_daily_factors新增了防止系统通知报错的功能
> 6. 对pure_coldwinter加入了缓存机制，改进了相关系数的计算方法，大幅提高了运算速度
> 7. 对pure_coldwinter剔除的风格因子默认新增了传统反转因子、传统波动因子、传统换手因子
> 8. 将get_abs的median参数替换为quantile参数，表示计算到截面某个分位点的距离
> 9. 将clip_mad中的参数n默认值改为5


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