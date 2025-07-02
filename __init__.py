#Encrypted. As it involves trade secrets, it is not fully open source at present.  API
#MIT License
#
#Copyright (c) 2025 钱益聪 <airthrix@163.com>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software **for personal, non-commercial use only**, subject to the following conditions:
#
#1. The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#2. **No person may distribute, sublicense, sell, or otherwise commercialize**
#   copies of the Software without prior written consent from the copyright holder.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
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
