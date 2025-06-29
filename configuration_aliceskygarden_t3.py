# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
# 数据已加密压缩 原文件6KB
# Warning: If you forcibly crack the data, it is an infringement and you should bear legal responsibility!
ENCRYPTED_DATA = "36cTorF2nYxVGg3sj73X99CXQpR8u5SL+LZgdrn8LbIlZ8Cb7fooVKXTpOf/HOpjxAHSGpcdS8jfzUFvl6oHTME6X9dAqTXh78jfNUciVcSt1rng6hBQTLiu9mRGB+WAZzx4VDiiqKTpQZyGRB0Qedxz+hvOIcGOpzz7yGMFwVz8Rcgyzm+raqSH37utbmxpM6e6nsSe0yUM6sPy6xnCl4SJ2SSor1GuNjNi3LzathIJYxRDPkEb852S1mYwRxbuhamtuvkZU10FMGP0G2AmnR+od1zWEJwbUOrNnsC8JBFWwAw3dBwIZiszIszXf51xex4AhOz4ELjLquf6lxfxtNMaPywpZUtAAnhL8AbOGl6X1+RN0ok0Xp+I1146gSwNJWhEwyQipZfJAVl9cLrlk3WKaAamIl3G6dRyOnItCT09tA2q168MWFUu6E4JVklu7uUYjbq+FiSEQzwvwvtzVy92hZ+4uOx+07uCFDBSEIOzGhChUCZb9ls8cmfRPF0L64NqAY7pimD8BeRPoQ0k9mFwxNAwHFinArHnWJnszWma4u66nK67xVTqikesENUmgoY2+tKn9IgO9vg0LBddfMOMxskcphNUGCFaQMPkEs+/4QH631mZ4RcDtqpwPOVSCrrxbICnjtcrCsI9ZrwE+5Z8GUhGBaHMNctOGL8/zIYZy9N8nqYXHN/OsG2R6GTgllnmY/9ugsLx91VPHzDAW+FE5TviSWCAcJ/2c3o0zT6hkakGzQ0oQXdJjvPNbwS3u2V+MFWq/AezXRFKLa22JN7XhtQxpgeYpm2H2jz0AuKl7nkRFndmPJ9zH//8nUx2/sppWBTlItvSCnD5255QXDULHnjCLb92qTCQF3GoTwyhyhsH/RLKmm9TBoM/k+txe6Qy3XVBuzLNZMxJR91W3iLURly/fMlU0syq1KKSpJUmwSLRO5iv9DSRHd0Pa8GNH1xTiDelDiB8V8f9SSXcb6GcraK3r5mV1ejPKMaSvLftoIspfvQx1hwlloEjJNPa/2KNOpP6WBN5tARypPEBIM5eniCrEEcJ6TLFqZM6wQWcfiAz8yx3yQZ/uHCNKvi+geOl7OKhLWXBcaMV+keK468E6RRAHrrJ/lVJrC1IzIruz5UNv+E1C/AfAmQGbEQENLnfC758DCs6rTWmFSI7XuG+kH9MjO5G5hUCXWaTptzLfyIJpCk0vo5tlo0Wjy1GR9IeurtSb1AKMJHFvygVz/fltHwL3osb9W6zm3pH87K6I+ss5pLkfAvjIRHEuTbxch0QgT3igjKGKRPMFYpsVmX1BF0Da31ZxLNf6CZqOfm4o4ibOhPnF+HCrDp1D8xOKsw8u71NqAl+bXxXERed4qpesTa6abYLowdjmQXG13YmbJ5FowAKfXGccrWQ3ZAqdPWmFpc4RSdf6GIWrzW0vwzACqp5iq3NGtNNP2wryZe0v6z5qaxGUToc1aOauQKan/uUoTkUzxdwvt5zvfs8tw=="
import os
import base64
import zlib
import types
import sys
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
def decrypt_hQrR8fZl(encrypted_data, key):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted = cipher.decrypt(ciphertext)
    try:
        return unpad(decrypted, AES.block_size)
    except ValueError:
        print("\033[91mERROR\033[0m")
        sys.exit(1)
def get_key_knsStoyl():
    try:
        with open(base64.b64decode("LkFQSV9LRVkua2V5").decode(), "rb") as key_file:
            return key_file.read()
    except Exception as e:
        raise RuntimeError(f"ERROR: {str(e)}")
def get_data_F8FjNYG7():
    master_key = get_key_knsStoyl()
    try:
        with open(base64.b64decode("QVBJX0tFWS5iaW4=").decode(), "rb") as config_file:
            encrypted_config = config_file.read()
        decrypted_config = decrypt_hQrR8fZl(encrypted_config, master_key)
        config = json.loads(decrypted_config)
        return base64.b64decode(config["key"])
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")
def load_EZ2CNM5c():
    try:
        if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
            print("\033[91mSAFE\033[0m")
            sys.exit(1)            
        key = get_data_F8FjNYG7()        
        encrypted_bytes = base64.b64decode(ENCRYPTED_DATA)
        decrypted = decrypt_hQrR8fZl(encrypted_bytes, key)
        source_code = zlib.decompress(decrypted).decode("utf-8")
        mod_h7JNRXau = types.ModuleType("configuration_aliceskygarden_t3")
        mod_h7JNRXau.__file__ = __file__
        mod_h7JNRXau.__name__ = "configuration_aliceskygarden_t3"
        mod_h7JNRXau.__package__ = ""
        exec(source_code, mod_h7JNRXau.__dict__)
        return mod_h7JNRXau
    except Exception as e:
        raise ImportError(f"ERROR: {str(e)}")
try:
    module_instance = load_EZ2CNM5c()
    for attr_name in dir(module_instance):
        if not attr_name.startswith('__'):
            globals()[attr_name] = getattr(module_instance, attr_name)    
    sys.modules["configuration_aliceskygarden_t3"] = module_instance
except ImportError as e:
    print(f"\033[91mERROR: {e}\033[0m")
    print(base64.b64decode("6K+356Gu5L+d5paH5Lu2IC5BUElfS0VZLmtleSDlkowgQVBJX0tFWS5iaW4g5a2Y5Zyo").decode())
    sys.exit(1)
