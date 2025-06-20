# Encrypted. As it involves trade secrets, it is not fully open source at present.  API

from typing import TYPE_CHECKING
from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .configuration_aliceskygarden_t3 import *
    from .modeling_aliceskygarden_t3 import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
