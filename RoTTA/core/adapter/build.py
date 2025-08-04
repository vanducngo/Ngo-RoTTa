from RoTTA.core.adapter.tent_multilabel import TentMultiLabel
from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .rotta_multiple_labels import RoTTA_MultiLabels


def build_adapter(cfg) -> type(BaseAdapter):
    adapterName = cfg.ADAPTER.NAME
    print(f'Using adapter: {adapterName}')
    if adapterName == "rotta":
        return RoTTA
    elif adapterName == "rotta_multilabels":
        return RoTTA_MultiLabels
    elif adapterName == 'tent':
        return TentMultiLabel
    # elif adapterName == 'cotta':
    #     return CoTTAMultiLabel # (Ví dụ cho tương lai)
    else:
        raise NotImplementedError("Implement your own adapter")

