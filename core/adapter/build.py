from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .rotta_multilabel_adapter import RoTTAMultiLabel


def build_adapter(cfg) -> type(BaseAdapter):
    adapterName = cfg.ADAPTER.NAME
    if adapterName == "rotta":
        return RoTTA
    elif adapterName == "rotta_multilabels":
        return RoTTAMultiLabel
    else:
        raise NotImplementedError("Implement your own adapter")

