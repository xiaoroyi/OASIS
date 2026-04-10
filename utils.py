import os
import io
import json
import sys
from pathlib import Path


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def load_hf_token(token_path: str | None = None) -> str | None:
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if env_token:
        token = env_token.strip()
        return token or None

    candidate = Path(token_path) if token_path else get_project_root() / "huggingface_token.txt"
    if not candidate.exists():
        return None

    token = candidate.read_text(encoding="utf-8").strip()
    return token or None

def modify_model_file(custom_modeling_opt_path, module_path, module_name):
    import importlib.util

    # custom_modeling_opt_path = "./models/modeling_opt_my.py"
    # spec = importlib.util.spec_from_file_location("modeling_opt", custom_modeling_opt_path)
    spec = importlib.util.spec_from_file_location(module_name, custom_modeling_opt_path)
    custom_modeling_opt = importlib.util.module_from_spec(spec)
    # sys.modules["transformers.models.opt.modeling_opt"] = custom_modeling_opt
    sys.modules[module_path] = custom_modeling_opt
    spec.loader.exec_module(custom_modeling_opt)
