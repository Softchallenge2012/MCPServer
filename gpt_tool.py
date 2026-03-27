import litellm
import os

# run litellm --config config_all.yaml --port 12732

async def chat(query: str, model: str, env_var: dict | None = None):
    # Determine credentials source: passed env dict vs server environment
    source = env_var if env_var is not None else os.environ
    
    api_base = source.get("LITELLM_PROXY_API_BASE")
    api_key = source.get("LITELLM_PROXY_API_KEY")

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": query}],
        api_base=api_base,
        api_key=api_key
    )
    return response.choices[0].message.content