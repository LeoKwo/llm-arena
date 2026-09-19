import os

from langchain_ollama.chat_models import ChatOllama


def _ensure_localhost_no_proxy():
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    hosts = [h.strip() for h in existing.split(",") if h.strip()]
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in hosts:
            hosts.append(host)
    value = ",".join(hosts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


_ensure_localhost_no_proxy()

PROVIDERS = {
    "ollama": {
        "label": "Local (Ollama)",
        "base_url": None,
        "env": [],
        "default_model": "qwen3.5:latest",
    },
    "zhipu": {
        "label": "Zhipu (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env": ["ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"],
        "default_model": "glm-4-plus",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "env": ["DEEPSEEK_API_KEY"],
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "default_model": "qwen-plus",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "env": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "default_model": "moonshot-v1-8k",
    },
}

LOCAL_MODELS = ["qwen3.5:latest", "qwen3:14b", "glm4:9b"]


def get_api_key(provider):
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    for name in spec["env"]:
        value = os.environ.get(name)
        if value:
            return value
    return None


def provider_status():
    status = {}
    for key, spec in PROVIDERS.items():
        if key == "ollama":
            status[key] = {
                "label": spec["label"],
                "configured": True,
                "default_model": spec["default_model"],
                "env": [],
            }
        else:
            status[key] = {
                "label": spec["label"],
                "configured": get_api_key(key) is not None,
                "default_model": spec["default_model"],
                "env": spec["env"],
            }
    return status


def build_chat_model(provider="ollama", model=None, temperature=0.1):
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown provider '{provider}'")
    if provider == "ollama":
        return ChatOllama(model=model or spec["default_model"], temperature=temperature)
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. "
            f"Set one of: {', '.join(spec['env'])}"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or spec["default_model"],
        base_url=spec["base_url"],
        api_key=api_key,
        temperature=temperature,
    )
