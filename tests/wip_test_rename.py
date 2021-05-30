import shutil
from pathlib import Path
import subprocess

import sinnfull

reporoot = Path(sinnfull.__file__).parent
repocopyroot = Path(__file__).parent/"sinnfull"

shutil.rmtree("scratch", ignore_errors=True)
shutil.copytree(reporoot, "scratch"/repocopyroot)
shutil.copy(reporoot.parent/"rename.py", "scratch/rename.py")

subprocess.run(["python", "scratch/rename.py", "myproject"])

# TODO: Add assertions for test substitutions
# - kernel name (sinn-full)
# - imports (sinnfull)
# - capitalized (Sinnfull)
# - file contents
# - file names
# - directory names
