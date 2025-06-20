# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
# 数据已加密
ENCRYPTED_DATA = r"IHS8oH4uLM2xPnO3KklvwTO1pbQDTKIF/FG2yz9vKFwBGOlVlfJkbdxTiTTk/7lfv4ZuIE+ASttJSYe8zAtO3fvddgS1fhfaBPmMFNJaoHClkZhs4IcxhxZyXx7b1o1kwGuM2mBkePNq7QPP/TyYEJo+nbDMG7MNGCjJZ6Nf0bYE4N1+kZ4T5qAW6pxoSoAAp8ajYOmouG7dbNhoa9ioYuY3QemopLE2I7efvtnaILgKd7NXgTwItJF2HEmXb3UnIUGQXKXcahHfO2qer4+RFF1YlwVATvh5fpU+2Ft6DLpyP82noQGfCXOvg6r1LG7jTrQNHTnPSHLxePjtzJeEq9EBG2FqfLXMwA0anGYp+S1MThgJi78ihgkok9MW3sq/obuhmiGRMaAS/+gLSBU93FdpCcJtcDjy42/gykQZGLIZry3UntCB34N0voaTcCSh9BhPhcpCIeTYyeRnqxrNq9iCnphHH8GUNdtC+N8L2rZEvoqX+KY98xhWcLzZGKQML309LxU0C1o5ynmsfqbD2IFFKOIutkaucmZh2xuWRw9zYQonECJy3OoAm3yAVBsl0KMBnYDihKvb0VkcSiYL1+n7vD0LxQVKPQqBqldOjy+tL2PkH90GOncAC9sXAhpc9bLUd9yNAljXde89RBRO2Uq2Zwd9ArSBKjRORvkrENnVU0TsvLies/NGWayHNe2aurCUmK4FTNZyWML7v2Iwex2kgRqMPE3VCa5sMthdmxyUwVIgi1Ay0kGPJeSp/tfU1fjlJD+4CA51hyWZobotJC96oSx8VIMDZrQWxL9DWmo2TWrp6g3U7YUdpJ12zbyW1js9KGWTWxBQk3arcldKXgF3T1Wzw9by1ef4GFwsPUqKm0DRyBn9WDHK6VuirWPuUnJij+KgTMnH99/7K52b7oJfmNpFBtUKpjJhMxzbUt40tEDtz1aYESNUNbRkVGIojA3QkRR8A5i5z9NacfKzlrqDn0Gjp63lAzjcQXnDzbQ2VLiF/TNArCTkmXWtYgKVKTjljdnDXCdkHbayVWfjihZxK0HfC4lpJh2L8VQjsr6k9t0QHXWoloSyWcfPlmrvjiAH+o/8ijWLEcv928l/Gvs9CbYWFMMgMJ603vXyjGtaBH+as5VLep6R7G1ePI3OSAIQwo8sM/HBBsMQ/FOfSdFb7yUnlPUzs/A+5NLHJKwsZme4aZfK1W9ro4lqeulKq52xs5QT9HfDnUhhCbuC01gFVFVZYeMUzfpf2Hzgv688h3kCqvmjiICvTFUsTRVRkHo0X2zmdm6PRvhTI2w2EXR+GrEVdCYwS+woDooDXh6VHJm5OpV3PaHxhfL3i2oCIeX1x+2utp+v813xxIdwQwqjqKrgrHGeLExry+p4L1SnoW0WGgR0LQ0HZ6dgNsuiIlGSu3oQBneq/Bymnrvo1+f/z4J6YjpPtEfaPJN1iJoW8yEPKkMg8HKAh+QT5n7+TRXERurizQacaMOEulgBOORNlEnq7lF7UnehYtuS9cvgYUH4tjvRaUJW4dzf4kADtK3+msR6Xuh7SkRuVUwAIEzLRd1LVORW1Fw8Dc9gigO8yw+SGjja0x21IYB23CexyKJ+aBS0G2bHKvhE+LzbgQ=="
# 数据已加密
import os
import base64
import zlib
import marshal
import types
import sys
import hashlib
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
        decompressed = zlib.decompress(decrypted)
        bytecode = marshal.loads(decompressed)
        module = types.ModuleType("configuration_aliceskygarden_t3")
        module.__file__ = __file__
        module.__name__ = "configuration_aliceskygarden_t3"
        module.__package__ = ""
        exec(bytecode, module.__dict__)
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
    print(f"please make sure file {MASTER_KEY_FILE} and {CONFIG_FILE} exist")
    sys.exit(1)
