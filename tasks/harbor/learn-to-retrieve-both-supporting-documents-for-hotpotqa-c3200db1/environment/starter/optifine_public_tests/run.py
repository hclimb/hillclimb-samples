import sys
from pathlib import Path

if not sys.flags.isolated:
    import os
    os.execv(sys.executable, [sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifiers.public_run import main

main()
