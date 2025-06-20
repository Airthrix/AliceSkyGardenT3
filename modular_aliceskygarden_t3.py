# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
# 数据已加密
ENCRYPTED_DATA = r"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"

import os
import base64
import zlib
import types
import sys
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
CONFIG_FILE = "API_KEY.bin"
MASTER_KEY_FILE = ".API_KEY.key"
def decrypt_data(encrypted_data, key):
    iv, ciphertext = encrypted_data[:16], encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size)

def get_master_key():
    try:
        with open(MASTER_KEY_FILE, "rb") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"ERROR: {str(e)}")

def get_data_key():
    master_key = get_master_key()    
    try:
        with open(CONFIG_FILE, "rb") as f:
            encrypted_config = f.read()       
        decrypted_config = decrypt_data(encrypted_config, master_key)
        config = json.loads(decrypted_config)
        return base64.b64decode(config["key"])
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")

def _secure_load():
    try:
        key = get_data_key()        
        encrypted_bytes = base64.b64decode(ENCRYPTED_DATA)        
        decrypted = decrypt_data(encrypted_bytes, key)        
        source_code = zlib.decompress(decrypted).decode("utf-8")        
        module = types.ModuleType("modular_aliceskygarden_t3")
        module.__file__ = __file__
        module.__name__ = "modular_aliceskygarden_t3"
        module.__package__ = ""        
        exec(source_code, module.__dict__)
        return module
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}") from e

try:
    _module = _secure_load()    
    for _name in dir(_module):
        if not _name.startswith('__'):
            globals()[_name] = getattr(_module, _name)   
    sys.modules["modular_aliceskygarden_t3"] = _module
    
except ImportError as e:
    print(f"\033[91mERROR: {e}\033[0m")
    print(f"please make sure {MASTER_KEY_FILE} 和 {CONFIG_FILE} exist")
    sys.exit(1)
