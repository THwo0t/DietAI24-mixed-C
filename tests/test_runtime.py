from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import hashlib
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import runtime


class RuntimeTests(unittest.TestCase):
    def test_interrupted_install_never_publishes_partial_final_file(self):
        with tempfile.TemporaryDirectory() as d:
            # Production ROOT is canonical even when macOS TMPDIR uses /var.
            root = Path(d).resolve()
            archive = root / 'runtime.zip'
            data = b'correct runtime bytes'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('data/value', data)
            spec = {'size_bytes': archive.stat().st_size, 'sha256': runtime.sha(archive),
                    'files': {'data/value': hashlib.sha256(data).hexdigest()}}
            original = runtime.shutil.copyfileobj
            calls = []
            def interrupted(source, dest):
                calls.append(1)
                if len(calls) == 2:
                    dest.write(b'partial')
                    raise OSError('simulated interruption')
                original(source, dest)
            with patch.object(runtime, 'ROOT', root), patch.object(runtime, 'specification', return_value=spec):
                with patch.object(runtime.shutil, 'copyfileobj', side_effect=interrupted):
                    with self.assertRaises(OSError):
                        runtime.install(archive)
                self.assertFalse((root / 'data/value').exists())
                runtime.install(archive)
                self.assertEqual((root / 'data/value').read_bytes(), data)
    def test_rejects_zip_path_escape_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('../escape', b'x')
            with zipfile.ZipFile(p) as z, self.assertRaises(ValueError):
                runtime.members(z, {'../escape': 'x'})

    def test_preserves_existing_different_runtime_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            p = root / 'runtime.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('data/value', b'original')
            (root / 'data').mkdir()
            target = root / 'data/value'
            target.write_bytes(b'user change')
            spec = {'size_bytes': p.stat().st_size, 'sha256': runtime.sha(p), 'files': {'data/value': '0' * 64}}
            with patch.object(runtime, 'ROOT', root), patch.object(runtime, 'specification', return_value=spec):
                with self.assertRaises(ValueError):
                    runtime.install(p)
            self.assertEqual(target.read_bytes(), b'user change')


if __name__ == '__main__':
    unittest.main()
