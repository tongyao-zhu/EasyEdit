"""Load exact upstream text-method files without eager multimodal imports.

The upstream package __init__ imports every editor and trainer. An isolated
private namespace avoids installing unrelated vision/API stacks. Algorithm
source files are neither copied nor patched; relative imports resolve to the
same checkout. This entry point is currently validated for FT only.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import types


def load_ft():
    root = Path(__file__).resolve().parents[1] / "easyeditor"
    prefix = "_ouro_easyedit_upstream"
    for suffix, path in [("", root), (".util", root / "util"),
                         (".models", root / "models")]:
        name = prefix + suffix
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            sys.modules[name] = module
        elif sys.modules[name].__path__ != [str(path)]:
            raise RuntimeError("Conflicting EasyEdit checkout already loaded")
    return importlib.import_module(prefix + ".models.ft.ft_main")
