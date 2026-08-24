"""Langfuse 可观测性集成：Trace / Token 成本 / 反馈闭环。

未配置 LANGFUSE_PUBLIC_KEY 时优雅降级（返回 None），开发环境零依赖。
"""
import os
from typing import Optional

# langfuse 4.x：`langfuse.callback` 已移除，现行公开路径为 langfuse.langchain
# （需安装 langchain 元包，langfuse.langchain.CallbackHandler 内部有 import langchain 版本探测）。
from langfuse.langchain import CallbackHandler

_instance: Optional[CallbackHandler] = None
_checked = False


def get_langfuse_handler() -> Optional[CallbackHandler]:
    global _instance, _checked
    if _checked:
        return _instance
    _checked = True
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        _instance = CallbackHandler()
    return _instance
