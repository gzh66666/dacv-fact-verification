
import os


def _setting(name, default=""):
    # Credentials and model selection come from the process environment; on Windows
    # the code also falls back to the user-level environment variables
    # (HKCU\Environment) when a variable is absent from the process environment.
    # No key literal is stored in this file.
    value = os.environ.get(name)
    if value is not None:
        return value
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
                return str(value)
        except (ImportError, OSError):
            pass
    return default


class ModelConfig:
    def __init__(self):
        model_id = _setting("DACV_MODEL_ID", "deepseek-v4-flash")
        api_key = _setting("DEEPSEEK_API_KEY", "")
        base_url = _setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        temperature = float(_setting("DACV_TEMPERATURE", "0.1"))

        self.configs = {
            "deepseek-flash": {
                "config_list": [{
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model_id,
                    "temperature": temperature,
                    "extra_body": {
                        "thinking": {"type": "disabled"}
                    },
                }],
                "cache_seed": 42,
            }
        }

    def get_config(self, model_name):
        if model_name == "deepseek-v4-flash":
            model_name = "deepseek-flash"
        config = self.configs.get(model_name)
        if not config:
            raise ValueError(f"Configuration for model {model_name} not found.")
        if not config["config_list"][0]["api_key"]:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        return config
