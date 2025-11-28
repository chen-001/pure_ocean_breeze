import sqlite3
import os

def save_factor_return(factor_name: str, neu_ret: float, save_dir: str, db_filename: str) -> bool:
    """
    保存因子收益率到 SQLite 数据库（线程安全）

    Args:
        factor_name: 因子名称
        neu_ret: 因子收益率
        save_dir: 保存目录（与图表同目录）
        db_filename: 数据库文件名（不带扩展名）

    Returns:
        是否保存成功
    """
    try:
        db_path = os.path.join(save_dir, f"{db_filename}.db")

        conn = sqlite3.connect(db_path)
        # 创建表（如果不存在）
        conn.execute('''
            CREATE TABLE IF NOT EXISTS factor_returns (
                factor_name TEXT PRIMARY KEY,
                neu_ret REAL NOT NULL
            )
        ''')
        # 使用 INSERT OR REPLACE 支持重复写入
        conn.execute(
            'INSERT OR REPLACE INTO factor_returns (factor_name, neu_ret) VALUES (?, ?)',
            (factor_name, neu_ret)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"保存因子收益率失败: {e}")
        return False

def get_all_factor_returns(save_dir: str, db_filename: str) -> dict:
    """
    获取所有因子收益率

    Args:
        save_dir: 保存目录（与图表同目录）
        db_filename: 数据库文件名（不带扩展名）

    Returns:
        字典：{factor_name: neu_ret}
    """
    try:
        db_path = os.path.join(save_dir, f"{db_filename}.db")

        if not os.path.exists(db_path):
            return {}

        conn = sqlite3.connect(db_path)
        cursor = conn.execute('SELECT factor_name, neu_ret FROM factor_returns')
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result
    except Exception as e:
        print(f"读取因子收益率失败: {e}")
        return {}
