"""Exact-pixel cache: unchanged tiles reuse results without approximate matching."""
from collections import OrderedDict
import hashlib
import threading

_CACHE=OrderedDict()
_LOCK=threading.Lock()
_LIMIT=4096


def recognize(reader, image, quality=None):
    key=(reader,image.shape,hashlib.blake2b(image.tobytes(),digest_size=16).digest(),quality is not None)
    with _LOCK:
        cached=_CACHE.get(key)
        if cached is not None:_CACHE.move_to_end(key)
    if cached is None:
        details={}
        if quality is None:
            name,score=reader.recognize_tile(image)
        else:
            try:name,score=reader.recognize_tile(image,quality=details)
            except TypeError:name,score=reader.recognize_tile(image)
        cached=(name,score,details)
        with _LOCK:
            _CACHE[key]=cached
            while len(_CACHE)>_LIMIT:_CACHE.popitem(last=False)
    if quality is not None:quality.update(cached[2])
    return cached[:2]


def clear_cache():
    with _LOCK:_CACHE.clear()
