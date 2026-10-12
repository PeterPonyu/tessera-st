"""Run a package CLI while denying author-workspace reads and all network calls."""
import os
import runpy
import sys
from pathlib import Path
package = Path(__file__).resolve().parent / 'relocated'
script = package / sys.argv[1]
sys.argv = [str(script)] + sys.argv[2:]

def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fsdecode(args[0]))
        if path.startswith('/home/zeyufu/Desktop/'):
            raise RuntimeError('Author workspace access blocked: ' + path)
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network calls blocked during saved-result verification')

sys.addaudithook(guard)
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
