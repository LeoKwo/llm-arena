import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama

# Load project-root .env once, before any provider/key lookup. Real environment
# variables take precedence over values defined in the file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


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
        "embedding_model": "qwen3-embedding:0.6b",
    },
    "zhipu": {
        "label": "Zhipu (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env": ["ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"],
        "default_model": "glm-4-plus",
        "embedding_model": "embedding-3",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "env": ["DEEPSEEK_API_KEY"],
        "default_model": "deepseek-v4-flash",
        "embedding_model": None,  # DeepSeek has no embeddings endpoint.
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "default_model": "qwen-plus",
        "embedding_model": "text-embedding-v4",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "env": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "default_model": "moonshot-v1-8k",
        "embedding_model": None,  # Kimi has no embeddings endpoint.
    },
}

LOCAL_MODELS = ["qwen3.5:latest", "qwen3:14b", "glm4:9b"]

# Embedding defaults (configurable via .env). DeepSeek has no embeddings API,
# so Zhipu's OpenAI-compatible embedding-3 is the default.
DEFAULT_EMBEDDING_PROVIDER = "zhipu"
DEFAULT_EMBEDDING_MODEL = "embedding-3"


def get_api_key(provider, override=None):
    """Resolve an API key for a provider.

    ``override`` (e.g. a key entered in the Web UI) takes precedence over the
    environment variables declared in ``PROVIDERS``.
    """
    if isinstance(override, str) and override.strip():
        return override.strip()
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    for name in spec["env"]:
        value = os.environ.get(name)
        if value:
            return value
    return None


def provider_status(overrides=None):
    overrides = overrides or {}
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
                "configured": get_api_key(key, overrides.get(key)) is not None,
                "default_model": spec["default_model"],
                "env": spec["env"],
            }
    return status


def build_chat_model(provider="ollama", model=None, temperature=0.1, api_key=None):
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown provider '{provider}'")
    if provider == "ollama":
        return ChatOllama(model=model or spec["default_model"], temperature=temperature)
    api_key = get_api_key(provider, api_key)
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


def embedding_config():
    """Resolve embedding settings from .env with sensible defaults."""
    provider = (
        os.environ.get("EMBEDDING_PROVIDER") or DEFAULT_EMBEDDING_PROVIDER
    ).strip().lower()
    model = (os.environ.get("EMBEDDING_MODEL") or "").strip()
    base_url = (os.environ.get("EMBEDDING_BASE_URL") or "").strip() or None
    api_key = (os.environ.get("EMBEDDING_API_KEY") or "").strip() or None
    if not model:
        if provider == DEFAULT_EMBEDDING_PROVIDER:
            model = DEFAULT_EMBEDDING_MODEL
        else:
            model = (PROVIDERS.get(provider) or {}).get("embedding_model")
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def build_embeddings(provider=None, model=None, base_url=None, api_key=None):
    """Build an embeddings client for FAISS memory.

    Defaults to Zhipu ``embedding-3`` because DeepSeek does not expose an
    embeddings endpoint. Set ``EMBEDDING_PROVIDER=ollama`` in .env to use a
    local embedding model instead.
    """
    conf = embedding_config()
    provider = (provider or conf["provider"]).strip().lower()
    model = model or conf["model"]
    base_url = base_url or conf["base_url"]
    api_key = api_key or conf["api_key"]

    if provider == "ollama":
        from langchain_ollama.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model=model or DEFAULT_EMBEDDING_MODEL)

    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown embedding provider '{provider}'")
    if not model:
        raise RuntimeError(
            f"No embedding model configured for provider '{provider}'. "
            f"Set EMBEDDING_MODEL in .env."
        )
    resolved_key = get_api_key(provider, api_key)
    if not resolved_key:
        raise RuntimeError(
            f"Missing API key for embedding provider '{provider}'. "
            f"Set EMBEDDING_API_KEY or one of: {', '.join(spec['env'])}"
        )

    resolved_base_url = (base_url or spec["base_url"] or "").rstrip("/")
    # OPENAI-compatible clients append "/embeddings" themselves; tolerate users
    # pasting the full endpoint URL (e.g. .../v4/embeddings) as the base.
    if resolved_base_url.endswith("/embeddings"):
        resolved_base_url = resolved_base_url[: -len("/embeddings")]

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        base_url=resolved_base_url,
        api_key=resolved_key,
        # OpenAI-compatible providers expect raw text, not tiktoken token IDs.
        check_embedding_ctx_length=False,
    )
