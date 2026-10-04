import logging
import os
import time
from logging.handlers import RotatingFileHandler

from windrecorder.const import DEBUGMODE_TRIGGER

logger = None

# 多进程共享同一日志文件时（main.py / record_screen.py / streamlit webui 都打开 wr.log），
# RotatingFileHandler.doRollover 的 os.rename 会因其他进程占用句柄而抛 PermissionError[WinError 32]，
# 导致本次轮转失败、触发轮转的那条日志被丢弃、错误刷屏。
# 此子类在轮转失败时跳过重命名并冷却一段时间，避免日志丢失；占用进程退出后下次轮转自然成功。
_ROLLOVER_COOLDOWN_S = 60


class _MultiProcessSafeRotatingFileHandler(RotatingFileHandler):
    _rollover_skip_until = 0.0

    def shouldRollover(self, record):
        if time.monotonic() < self._rollover_skip_until:
            return False
        return super().shouldRollover(record)

    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            # 其他进程占用 wr.log 使 os.rename 失败 — 跳过本次轮转，冷却避免每条日志都重试
            self._rollover_skip_until = time.monotonic() + _ROLLOVER_COOLDOWN_S
            # super() 已关闭 self.stream，重新打开以继续写入当前文件
            if self.stream is None:
                self.stream = self._open()


def get_logger(name, log_name="wr.log"):
    global logger
    if logger is not None:
        return logger

    log_dir = "cache\\logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(levelname)s - %(message)s")

    # 创建一个滚动文件处理器，每个日志文件最大大小为5M，保存5个旧日志文件
    rf_handler = _MultiProcessSafeRotatingFileHandler(
        os.path.join(log_dir, log_name),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    # rf_handler = TimedRotatingFileHandler(os.path.join(log_dir, log_name), when="d", interval=1, backupCount=7)
    # rf_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"  # 设置历史文件 后缀
    rf_handler.setFormatter(formatter)
    logger.addHandler(rf_handler)

    if os.path.exists(DEBUGMODE_TRIGGER):
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


"""
usage:

from windrecorder.logger import get_logger

logger = get_logger(__name__)
"""
