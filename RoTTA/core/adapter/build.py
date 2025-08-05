from RoTTA.core.adapter.cotta_multilabel import CoTTAMultiLabel
from RoTTA.core.adapter.tent_multilabel import TentMultiLabel
from RoTTA.core.adapter.norm_multilabel import NormMultiLabel
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
    elif adapterName == 'norm':
        return NormMultiLabel
    elif adapterName == 'cotta':
        return CoTTAMultiLabel
    else:
        raise NotImplementedError("Implement your own adapter")

