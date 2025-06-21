# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
# 数据已加密压缩 原文件6KB
ENCRYPTED_DATA = r"UMxLG9KEO9X1dGAh6BRecUmNXtPy3PBpCXUn8V4gQZrzCAwaHmdtC+2+hL4oUsz5nl55Sma5l/sP2yIjeJtk0UYjkiIJM+Gz9HYu1FEJJ0hhzNnwJVcJ30KPHfWAULUyJjdlF1WQsi/3IxLYx1tfG7+3prwd3Qf6hqUsfbk14LohvDRCjc/AE+wHJGqwXNNo99b9H/PVDh+rC55GRju1KkvGNBRpyge/ZGwT5MObTPepqFg7NMUUh/u3NaeYScmFznL7qdMCcuslSrXNb3b6/4YujOn526yBhJ4D4i6vaulH3eFbcwRoy9T82Ls0p5Ao9N6au+J4NdLfkoSYjE9+IDHgBJLJ/1OOkEI6zseGSAFSXAZmtIBP9gwPeNCwTGF1dfzHzLagx7Uka/Leog43HCOxKcAe5J+MsUvZ/N9LRb1IFPZCmnBfgUX6jFqc1kJJH8SICPjTBCK9IK2gOtlFfhIsFVg3+Oj9NZ/4JIVDNod8wloEprvm6ZlRxUJ+rpF24mryzDBWiBArsjcslgR6yYaUrNCZfGE2usvogQiufOjaLJbL7I0t2n8X8gP4+yViNp6DtQRVVgpi5jOPhs3v2/mv0JIgazCe3o1xH3KP0NnS96b4iqPSvryJZy++OY/ilOu+eNwCu0ueMeHxRNGAFGwEUOEbqpPkEzMlRquFdqGopr1GE7GdWYrGO4nm+qWfT40lFpbCD3m1G7lgfyLRJsQNkISks4j6opEc1joP4OD4FN5CnHtCjhbNNOf5zabKF5yD47q78OxQewngfmwTJL1t8xxNgvdDlhV+6xxPehM+KAT2RwtWdw3FOLILQ4o07pY8YEFK5o0Z5+0BMLIava4mBRzzBKo+F6GU1YhZ2b+1gG7RKT83hu3wUm2eFh9VA1zwJ1s44Jzt8iv6PdlhZ4Q12Pq+NKqERiaIRLB678uyJnK37LH6tqp/yY//PrJ1+FNBpA8+iJFP+X2O8i8NYzLGlSAHsFHNGJoXcR8WzXmVbfBXp1ye4QF6UaCvwWpIGvYQH25tbmgsKguj8+cY8dZIFYXR9szNpDw5WTOoCgzAo1n6LD6U8jGxO4dRbbF3gCvnkF8M88kIP3EVYZRvUuY4uPhduGBM/UU50ZZ48QVixBzA/kQUup6MWoXZG3wqgjvLnvme1IIpstkl1FfG4cmIwVHK41WU3wFyKPGIrFgZJorIgGGIX109M4M9rEIBMjEpDUq7KwHqntNiu3Vv5eIUNIGY9Nn2ceZNSF7mY2mAlZJBKCxVBiluOEWMwafk0YyBoh55F4l0tHjKEcmxuz0+ndgHDjx7UiFjUGilRD58UFwOyEiuYPQLz5pfJUcs222rdO3AzpfVMrscqBc9erlw/2mFR9/cbaVn24lFGCukzApcthUuDbM9ekbPGEHXmT8D0wHYE3lA/6WTLWXj/ypI9XBa96ExKv7J142RHmfeY8pLrdY/Q4FeaISdUysC6VbGqZei++pW8uOPppuvRg=="

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
        module = types.ModuleType("configuration_aliceskygarden_t3")
        module.__file__ = __file__
        module.__name__ = "configuration_aliceskygarden_t3"
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
    sys.modules["configuration_aliceskygarden_t3"] = _module
    
except ImportError as e:
    print(f"\033[91mERROR: {e}\033[0m")
    print(f"please make sure {MASTER_KEY_FILE} 和 {CONFIG_FILE} exist")
    sys.exit(1)
