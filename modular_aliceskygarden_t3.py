# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
# 数据已加密压缩 原文件8KB
# Warning: If you forcibly crack the data, it is an infringement and you should bear legal responsibility!
ENCRYPTED_DATA = "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"
import os
import base64
import zlib
import types
import sys
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
def decrypt_TVfJuGbr(encrypted_data, key):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted = cipher.decrypt(ciphertext)
    try:
        return unpad(decrypted, AES.block_size)
    except ValueError:
        print("\033[91mERROR\033[0m")
        sys.exit(1)
def get_key_xIrhpFVy():
    try:
        with open(base64.b64decode("LkFQSV9LRVkua2V5").decode(), "rb") as key_file:
            return key_file.read()
    except Exception as e:
        raise RuntimeError(f"ERROR: {str(e)}")
def get_data_B8zZSSYT():
    master_key = get_key_xIrhpFVy()
    try:
        with open(base64.b64decode("QVBJX0tFWS5iaW4=").decode(), "rb") as config_file:
            encrypted_config = config_file.read()
        decrypted_config = decrypt_TVfJuGbr(encrypted_config, master_key)
        config = json.loads(decrypted_config)
        return base64.b64decode(config["key"])
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")
def load_txFWSVwL():
    try:
        if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
            print("\033[91mSAFE\033[0m")
            sys.exit(1)            
        key = get_data_B8zZSSYT()        
        encrypted_bytes = base64.b64decode(ENCRYPTED_DATA)
        decrypted = decrypt_TVfJuGbr(encrypted_bytes, key)
        source_code = zlib.decompress(decrypted).decode("utf-8")
        mod_nMEQ1g2h = types.ModuleType("modular_aliceskygarden_t3")
        mod_nMEQ1g2h.__file__ = __file__
        mod_nMEQ1g2h.__name__ = "modular_aliceskygarden_t3"
        mod_nMEQ1g2h.__package__ = ""
        exec(source_code, mod_nMEQ1g2h.__dict__)
        return mod_nMEQ1g2h
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")
try:
    module_instance = load_txFWSVwL()
    for attr_name in dir(module_instance):
        if not attr_name.startswith('__'):
            globals()[attr_name] = getattr(module_instance, attr_name)    
    sys.modules["modular_aliceskygarden_t3"] = module_instance
except ImportError as e:
    print(f"\033[91mERROR: {e}\033[0m")
    print(base64.b64decode("6K+356Gu5L+d5paH5Lu2IC5BUElfS0VZLmtleSDlkowgQVBJX0tFWS5iaW4g5a2Y5Zyo").decode())
    sys.exit(1)
