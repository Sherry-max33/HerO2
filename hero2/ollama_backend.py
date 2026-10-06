import json
import os
import time
import urllib.error
import urllib.request

# Generation stages go through OpenRouter. No local model weights.
# Verdict is general Qwen3-32B, pinned to DeepInfra FP8, not humane-lab/Qwen3-32B-AWQ-HerO.
# Qwen3-8B is hosted only by Alibaba on OpenRouter. Llama 3.1 8B is pinned to DeepInfra.
_OLLAMA_TAGS = {
    "Qwen/Qwen3-8B": "qwen3:8b",
    "Qwen/Qwen3-32B": "qwen3:32b",
    "humane-lab/Qwen3-32B-AWQ-HerO": "qwen3:32b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3:8b",
    "humane-lab/Meta-Llama-3.1-8B-HerO": "llama3.1:8b",
}

_OPENROUTER_ROUTES = {
    "qwen3:8b": {
        "model": "qwen/qwen3-8b",
        "provider": None,
        "no_think": True,
    },
    "qwen3:32b": {
        "model": "qwen/qwen3-32b",
        "provider": {"only": ["deepinfra/fp8"], "allow_fallbacks": False},
        "no_think": True,
    },
    "llama3.1:8b": {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        "no_think": False,
    },
    "llama3:8b": {
        "model": "meta-llama/llama-3-8b-instruct",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        "no_think": False,
    },
    "qwen/qwen3.8-27b:free": {
        "model": "qwen/qwen3.8-27b:free",
        "provider": None,
        "no_think": True,
    },
}
_OPENROUTER_MODELS = set(_OPENROUTER_ROUTES) | {
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "qwen/qwen3-8b",
    "qwen/qwen3-32b",
    "humane-lab/Qwen3-32B-AWQ-HerO",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "humane-lab/Meta-Llama-3.1-8B-HerO",
    "qwen/qwen3.8-27b:free",
}
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_EMBED_MODEL = "thenlper/gte-base"
_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
_GTE_MODEL_NAMES = {
    "Alibaba-NLP/gte-base-en-v1.5",
    "thenlper/gte-base",
}

_TOKENIZER_IDS = {
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3:32b": "Qwen/Qwen3-32B",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "humane-lab/Qwen3-32B-AWQ-HerO": "Qwen/Qwen3-32B",
    "qwen/qwen3.8-27b:free": "Qwen/Qwen3-8B",
}


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv_file():
    path = os.path.join(repo_root(), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def ollama_base_url():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return host


def uses_embedding_api(model_name):
    return model_name in _GTE_MODEL_NAMES


def embed_texts(texts, batch_size=32):
    import numpy as np

    load_dotenv_file()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Put OPENROUTER_API_KEY in " + os.path.join(repo_root(), ".env")
        )
    vectors = [None] * len(texts)
    total_cost = 0.0
    for start in range(0, len(texts), batch_size):
        chunk = [text[:2000] for text in texts[start : start + batch_size]]
        body = {
            "model": _EMBED_MODEL,
            "input": chunk,
            "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        }
        request = urllib.request.Request(
            _EMBED_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter embeddings failed: {exc.code} {detail}") from exc
        for item in data["data"]:
            vectors[start + item["index"]] = item["embedding"]
        usage = data.get("usage") or {}
        total_cost += float(usage.get("cost") or 0.0)
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        raise RuntimeError(f"OpenRouter embeddings missing vectors at {missing[:5]}")
    print(f"OpenRouter embeddings: model={_EMBED_MODEL} n={len(texts)} cost={total_cost:.6f}")
    return np.asarray(vectors, dtype=np.float32)


def to_ollama_model(model):
    return _OLLAMA_TAGS.get(model, model)


def uses_openrouter(model):
    return model in _OPENROUTER_MODELS or to_ollama_model(model) in _OPENROUTER_MODELS


def chat_user_content(prompt):
    marker = "<|im_start|>user"
    if marker in prompt and "<|im_end|>" in prompt:
        rest = prompt[prompt.find(marker) + len(marker):].lstrip("\n")
        end = rest.find("<|im_end|>")
        if end != -1:
            return rest[:end]
    return prompt


def tokenizer_source(model):
    if model in _TOKENIZER_IDS:
        return _TOKENIZER_IDS[model]
    ollama_model = to_ollama_model(model)
    if ollama_model in _TOKENIZER_IDS:
        return _TOKENIZER_IDS[ollama_model]
    return model


class _Llama31ChatTemplate:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        pieces = ["<|begin_of_text|>"]
        for message in messages:
            pieces.append(
                f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                f"{message['content']}<|eot_id|>"
            )
        if add_generation_prompt:
            pieces.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(pieces)


class SamplingParams:
    def __init__(
        self,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=16,
        n=1,
        stop=None,
        logprobs=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        skip_special_tokens=True,
        **kwargs,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.logprobs = logprobs
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.skip_special_tokens = skip_special_tokens


class CompletionOutput:
    def __init__(self, text):
        self.text = text
        self.logprobs = None


class RequestOutput:
    def __init__(self, texts):
        self.outputs = [CompletionOutput(text) for text in texts]


class LLM:
    def __init__(self, model, max_model_len=4096, **kwargs):
        self.model_name = model
        self.ollama_model = to_ollama_model(model)
        self.hf_tokenizer_id = tokenizer_source(model)
        self.max_model_len = max_model_len
        self.openrouter = uses_openrouter(model)
        self.openrouter_route = _OPENROUTER_ROUTES.get(self.ollama_model)
        self._tokenizer = None
        if self.openrouter:
            route = self.openrouter_route or {}
            print(
                "OpenRouter backend: model="
                + route.get("model", self.ollama_model)
                + f" tokenizer={self.hf_tokenizer_id}"
            )
        else:
            print(
                f"Ollama backend: model={self.ollama_model} "
                f"tokenizer={self.hf_tokenizer_id} num_ctx={self.max_model_len} "
                f"host={ollama_base_url()}"
            )

    def get_tokenizer(self):
        if self._tokenizer is None:
            import transformers

            try:
                self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.hf_tokenizer_id,
                    trust_remote_code=True,
                )
            except Exception as exc:
                if "Llama-3.1" not in self.hf_tokenizer_id and "llama3.1" not in self.ollama_model:
                    raise
                print(
                    "Llama tokenizer is gated, using the Llama 3.1 chat template locally: "
                    + type(exc).__name__
                )
                self._tokenizer = _Llama31ChatTemplate()
        return self._tokenizer

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        return [RequestOutput(self._sample(prompt, sampling_params)) for prompt in prompts]

    def _sample(self, prompt, sampling_params):
        sample_count = sampling_params.n or 1
        return [self._generate_once(prompt, sampling_params) for _ in range(sample_count)]

    def _generate_once(self, prompt, sampling_params):
        if self.openrouter:
            return self._openrouter_once(prompt, sampling_params)
        return self._ollama_once(prompt, sampling_params)

    def _openrouter_once(self, prompt, sampling_params):
        load_dotenv_file()
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Put OPENROUTER_API_KEY in "
                + os.path.join(repo_root(), ".env")
            )
        route = self.openrouter_route
        if route is None:
            raise RuntimeError("No OpenRouter route for " + self.ollama_model)
        user_content = chat_user_content(prompt)
        if route["no_think"] and "/no_think" not in user_content:
            user_content = user_content.rstrip() + "\n/no_think"
        body = {
            "model": route["model"],
            "messages": [{"role": "user", "content": user_content}],
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
        }
        if route["provider"] is not None:
            body["provider"] = route["provider"]
        if route["no_think"]:
            body["reasoning"] = {"enabled": False}
        if sampling_params.top_k is not None and sampling_params.top_k >= 0:
            body["top_k"] = sampling_params.top_k
        if sampling_params.min_p:
            body["min_p"] = sampling_params.min_p
        if sampling_params.stop:
            body["stop"] = sampling_params.stop
        if sampling_params.frequency_penalty:
            body["frequency_penalty"] = sampling_params.frequency_penalty
        if sampling_params.presence_penalty:
            body["presence_penalty"] = sampling_params.presence_penalty

        payload = json.dumps(body).encode("utf-8")
        data = None
        for attempt in range(20):
            request = urllib.request.Request(
                _OPENROUTER_URL,
                data=payload,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                in_flight = exc.code == 402 and "in_flight" in detail
                retryable = (
                    exc.code in (429, 500, 502, 503, 504) or in_flight
                ) and attempt < 19
                if retryable:
                    if in_flight:
                        wait = 120
                    elif exc.code == 429:
                        wait = min(180, 30 * (attempt + 1))
                    else:
                        wait = min(60, 2 ** attempt)
                    print(
                        f"OpenRouter {exc.code}, retry in {wait}s ({attempt + 1}/20)",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenRouter request failed: {exc.code} {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < 19:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise RuntimeError(f"Cannot reach OpenRouter at {_OPENROUTER_URL}.") from exc

        provider = str(data.get("provider") or "")
        pinned = route["provider"]
        if pinned and pinned.get("only") and provider:
            expected = pinned["only"][0].split("/")[0]
            if not provider.lower().startswith(expected):
                raise RuntimeError(
                    f"OpenRouter routed to {provider}, expected {expected}."
                )
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content and message.get("reasoning"):
            raise RuntimeError("Qwen3-32B returned reasoning text with empty content.")
        return content

    def _ollama_once(self, prompt, sampling_params):
        options = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "num_predict": sampling_params.max_tokens,
            "num_ctx": self.max_model_len,
        }
        if sampling_params.top_k is not None and sampling_params.top_k >= 0:
            options["top_k"] = sampling_params.top_k
        if sampling_params.min_p:
            options["min_p"] = sampling_params.min_p
        if sampling_params.stop:
            options["stop"] = sampling_params.stop
        if sampling_params.frequency_penalty:
            options["frequency_penalty"] = sampling_params.frequency_penalty
        if sampling_params.presence_penalty:
            options["presence_penalty"] = sampling_params.presence_penalty

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": "30m",
            "options": options,
        }
        request = urllib.request.Request(
            ollama_base_url() + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed for {self.ollama_model}: {exc.code} {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama at "
                f"{ollama_base_url()}. Start it with `ollama serve`, then "
                f"`ollama pull {self.ollama_model}`."
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Ollama error for {self.ollama_model}: {data['error']}")
        return data.get("response", "")
