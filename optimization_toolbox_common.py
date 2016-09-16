import json, os
from contextlib import contextmanager
import tempfile

_json_pretty = dict(sort_keys=True, indent=4, separators=(',',': '))

@contextmanager
def atomic_open_for_write(filename, mode='w+', **kwargs):
    tmpname = None
    try:
        with tempfile.NamedTemporaryFile(
                mode=mode, delete=False,
                dir=os.path.dirname(filename), **kwargs) as handle:
            tmpname = handle.name
            yield handle
        os.rename(tmpname, filename)
    finally:
        try:
            if tmpname is not None:
                os.remove(tmpname)
        except (IOError, OSError):
            pass

def atomic_json_write(filename, obj):
    with atomic_open_for_write(filename) as fp:
        json.dump(obj, fp, **_json_pretty)

