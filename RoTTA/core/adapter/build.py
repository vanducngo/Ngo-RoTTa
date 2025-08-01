from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .rotta_multiple_labels import RoTTA_MultiLabels


def build_adapter(cfg) -> type(BaseAdapter):
    adapterName = cfg.ADAPTER.NAME
    if adapterName == "rotta":
        return RoTTA
    elif adapterName == "rotta_multilabels":
        return RoTTA_MultiLabels
    else:
        raise NotImplementedError("Implement your own adapter")

