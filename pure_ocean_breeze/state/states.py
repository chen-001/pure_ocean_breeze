"""
一些默认的参数
"""

__updated__ = "2023-07-13 12:18:16"

STATES = {
    "NO_LOG": False,
    "NO_COMMENT": False,
    "NO_SAVE": True,
    "NO_PLOT": False,
    "START": 20140101,
    "db_host": "127.0.0.1",
    "db_port": 3306,
    "db_user": "root",
    "db_password": "Kingwila98",
}

COMMENTS_WRITER=None
NET_VALUES_WRITER=None
ON_PAPER=False
MOON_START=None
MOON_END=None

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False