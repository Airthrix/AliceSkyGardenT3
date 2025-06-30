# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
# 数据已加密压缩 原文件6KB
# Warning: If you forcibly crack the data, it is an infringement and you should bear legal responsibility!
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
ENCRYPTED_DATA = "1bOO3ZnkkT53WVcXOmZpO4J+at2vbi44zGnEWolWmoXVyz0paTup/vsnAeOKMg2yZ6Pb+6XAa66LuzzGwamQn9SV8HJsq1vo2iZYHfmiz8ooSfiKt3nD3F3VeiZBGqDSh4rmuHQwhn85nERj66h24l8amYcEgXzwqnArF3XluTF/lvaBfxVKexgUZP/WzKPxVmn8ZiPymLKHQeoUu3Y1p3fDuKZ8IqDCY3kbxgOotWgG2/KpLYvbq88pIgC/cjVaZo9ocR4FwoZDDZ12ZExeK59b4PzJokK3Ku5qNhpGmCCqm+sNtRKR0YYtSTtm/60iwvRAhC8oE7cC6edwGrUU6gxQEys0fwJJHHLrKOgN9rA5AwwNJH0L+NSycXoAYPvVJ+vZ7hRuTU3XGZXETzV28NZXlz3di8DS+pO446z4ZUmzs0Xe/YwWxca1C1Lq8n5vBANz8tYM6zySQ0yjK7bbPWm3Z+/bo5SGtD4VedW4o+Q5lCZuxMkb1C8SU0c5LSMFeR2XXj2QFVb454aOMZisYvET0pLoRDS2M8cTUGxhzWKPD9q+O/ClN1kq4letapVwi1/DToiblgdZP089oKyB7w2LFUOXMD8Yeioq43Wck6f0JmcrDPrsOO79NGRBJg28SwgKpYBmpMnyaESn/btTJdsaI4hHMFcnmfrEMy04KEavA8+HgrjwezXz3vvbBi22suVoiF2R54+e4y9xRZYJBXMVVukp1npKrE+LGGKnBh1BfxKQAqVO+8JoV9YHuMktHg89jbWZwUJb5rVHjhreHQOgZW0F1VgskmHkiL1N4t9ccm6JRodCR2nLd+6oDRtEhjwuowY/4GjRv9XXpTi9XugDVZMUPys4y/xzxzAsecg9klWw75JnQMLmllDUzI89qBUx20ocEQd8M0B//ZyKxgfpMvXLRKLt+4rB9rmUoN29I1aLPqxaI/HlzzybTq1JxgWysVLX78gn5p+BJ2/P5vqOSDoCvqLKswNLBH7Wwm9QGzO8CE0hKvfqFF4pFaH3ziqZU753HiMSa1gZEAk1IjuGMl7DJMJC1dchVIQAd3wmU3NmqiHhaZK/aQZvCdz4k/15aisBOFxptVmN1+Cuf7JW924dRp8dKLBPwZUIn2UWK9hLS+pqdrSjA8YStkvulUwj28Yikw2ZH6LCd1AYZfaQZw5z6i0CxBdgLJJvAINVakVN5o9sTgGADdxyAgs5EJecnfv62vagVw/cFM3zAet1E2cpKcEAaD1fvC1WxdKqs1pzrPSHwSo8yPSDwhhl5/QP2xBdJSfUvOIr6lmfC8PDPAQeIjgUTIhunWdQynUx2h9QcKYbnuOA2sJWch6IY3yzTIJuL+vsxBs0w4/Lk/VqeoEzvSoIE4uS2zHAA5JHc4rCTDWYZeU1OAPPFbp/J/c1wGD8yJj5w/t9O30KURVrndRd5bgd2J+xkApX+mHTUxf9tnn4W9I325fKCct5keWurBoiNt4q1m2JVdaWO2BzZTRqHng3WfUz/t3Kn8eP4Tey4iXQYYJ0EX5f4rfgtuwBE7pgnU3LVlnMj1dGwux113pPE9cQZrLvfqkiO6u0jJ4XMKVPAiSADhRjkzikCGr1eIj/OYS+gcdgWcQeXmSSQqRots8wSbSMrj7SEPDrVRRxG5izCD9lOfA7QDnXqN/NMe55Ku2XBpi6LWp4byHNFeL82s/OvdpccUJxfbeREorDX5eCWZuMTyB3SFOBE3+weYqLarTIA7ZIseE4QfSTRY6dcsPPde4ur4wksXSoZvV55Z1Sv6gvKRHw/kIOPUQTc1WYOzxiami0ssan5Cwfndl2MBVUjANtJVXHcbVSTKadmaAtIc3DrdvNCBOcP0HSI4OCRUbjhgIZ7oQtAnFZx6CbfSkSAH+5ymgpNEqmMVBHLVhUJDMUfawgG73H"
import os
import base64
import zlib
import types
import sys
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
def decrypt_Xdone20r(encrypted_data, key):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted = cipher.decrypt(ciphertext)
    try:
        return unpad(decrypted, AES.block_size)
    except ValueError:
        print("\033[91mERROR\033[0m")
        sys.exit(1)
def get_key_ieu0b0xx():
    try:
        with open(base64.b64decode("LkFQSV9LRVkua2V5").decode(), "rb") as key_file:
            return key_file.read()
    except Exception as e:
        raise RuntimeError(f"ERROR: {str(e)}")
def get_data_JF6TBRMn():
    master_key = get_key_ieu0b0xx()
    try:
        with open(base64.b64decode("QVBJX0tFWS5iaW4=").decode(), "rb") as config_file:
            encrypted_config = config_file.read()
        decrypted_config = decrypt_Xdone20r(encrypted_config, master_key)
        config = json.loads(decrypted_config)
        return base64.b64decode(config["key"])
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")
def load_TuhJIdza():
    try:
        if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
            print("\033[91mSAFE\033[0m")
            sys.exit(1)            
        key = get_data_JF6TBRMn()        
        encrypted_bytes = base64.b64decode(ENCRYPTED_DATA)
        decrypted = decrypt_Xdone20r(encrypted_bytes, key)
        source_code = zlib.decompress(decrypted).decode("utf-8")
        mod_ewEQNLJy = types.ModuleType("configuration_aliceskygarden_t3")
        mod_ewEQNLJy.__file__ = __file__
        mod_ewEQNLJy.__name__ = "configuration_aliceskygarden_t3"
        mod_ewEQNLJy.__package__ = ""
        exec(source_code, mod_ewEQNLJy.__dict__)
        return mod_ewEQNLJy
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")
try:
    module_instance = load_TuhJIdza()
    for attr_name in dir(module_instance):
        if not attr_name.startswith('__'):
            globals()[attr_name] = getattr(module_instance, attr_name)    
    sys.modules["configuration_aliceskygarden_t3"] = module_instance
except ImportError as e:
    print(f"\033[91mERROR: {e}\033[0m")
    print(base64.b64decode("6K+356Gu5L+d5paH5Lu2IC5BUElfS0VZLmtleSDlkowgQVBJX0tFWS5iaW4g5a2Y5Zyo").decode())
    sys.exit(1)
