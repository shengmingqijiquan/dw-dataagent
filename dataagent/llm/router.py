"""模型路由层（Tiered Model Stack）。

生产对标：大厂模型网关的简化实现——按任务类型路由到不同模型，
成本与质量分层；provider 抽象隔离 API 与本地部署差异，
生产私有化部署（vLLM）时业务代码零改动。
"""
import os

from langchain_core.language_models import BaseChatModel

from dataagent.config import Settings


class LLMRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, BaseChatModel] = {}

    def get_model(self, task: str) -> BaseChatModel:
        """按任务类型返回模型（实例缓存）。task ∈ task_parse/sql_generate/critic_review"""
        routing = self.settings.llm.routing
        provider_name = routing.get(task) or routing.get("sql_generate") or "deepseek"
        return self._get_provider(provider_name)

    def get_model_with_fallback(self, task: str) -> BaseChatModel:
        """主模型不可用时降级到 fallback provider。"""
        routing = self.settings.llm.routing
        provider_name = routing.get(task) or routing.get("sql_generate") or "deepseek"
        try:
            return self._get_provider(provider_name)
        except Exception:
            fallback = routing.get("fallback")
            if fallback and fallback != provider_name:
                return self._get_provider(fallback)
            raise

    def _get_provider(self, name: str) -> BaseChatModel:
        if name in self._cache:
            return self._cache[name]
        provider = self.settings.llm.providers[name]
        if provider.type == "api":
            api_key = os.environ.get(provider.api_key_env)
            if not api_key:
                raise ValueError(
                    f"环境变量 {provider.api_key_env} 未设置，无法使用 {name}")
            from langchain_deepseek import ChatDeepSeek
            model = ChatDeepSeek(
                model=provider.model,
                api_base=provider.base_url,
                api_key=api_key,
                temperature=0.1,
            )
        elif provider.type == "local":
            from langchain_ollama import ChatOllama
            model = ChatOllama(
                model=provider.model,
                base_url=provider.base_url,
                temperature=0.1,
            )
        else:
            raise ValueError(f"未知 provider 类型: {provider.type}")
        self._cache[name] = model
        return model
