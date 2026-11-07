"""Run both unchanged-method and portable checks with a canonical temp root."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    # On macOS /var is an alias of /private/var. The archived provider tests
    # patch ROOT with the temporary path, while production ROOT is resolved.
    # Give the child interpreters its physical path; keep original tests intact.
    env = dict(os.environ, TMPDIR=str(Path(tempfile.gettempdir()).resolve()))
    for directory in ('tests', 'optimization_sol_20261102/mae_v2/c_stage_split_20261105'):
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                        '-s', directory, '-p', 'test_*.py', '-v'],
                       cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
