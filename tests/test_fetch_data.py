import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fetch_data as fetch


def asa_row(data):
    digest = hashlib.sha256(data).hexdigest()
    return {
        "image_id": f"asa24-{digest}",
        "dataset": "asa24",
        "sha256": digest,
        "source_path": "example.png",
        "path": f"data/images/development/asa24/{digest}.png",
    }


class FetchDataTests(unittest.TestCase):
    def test_manifest_counts(self):
        for cohort, expected in (
            ("historical", (663, 843)),
            ("development", (3000, 3000)),
            ("external50", (50, 50)),
        ):
            images, samples = fetch.read_manifest(fetch.ROOT, cohort)
            self.assertEqual((len(images), len(samples)), expected)

    def test_zip_selects_only_allowed_member_and_checks_bytes(self):
        data = b"frozen bytes"
        row = asa_row(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            zip_path = root / "source.zip"
            with ZipFile(zip_path, "w") as output:
                output.writestr("wrong/example.png", b"wrong bytes")
                output.writestr("right/example.png", data)
                output.writestr("protected.png", b"outside allowlist")
            with ZipFile(zip_path) as archive:
                self.assertEqual(fetch.materialize([row], root, None, archive, False), 0)
            self.assertEqual((root / row["path"]).read_bytes(), data)
            self.assertEqual(fetch.materialize([row], root, None, None, False), 0)
            self.assertFalse((root / "protected.png").exists())

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zip"
            with ZipFile(path, "w") as output:
                output.writestr("../example.png", b"unsafe path")
            with ZipFile(path) as archive, self.assertRaises(ValueError):
                fetch.zip_index(archive, {"example.png"})

    def test_bad_local_bytes_do_not_create_destination(self):
        row = asa_row(b"frozen bytes")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "example.png").write_bytes(b"wrong bytes")
            self.assertEqual(fetch.materialize([row], root, root, None, False), 1)
            self.assertFalse((root / row["path"]).exists())

    def test_foreign_url_and_linked_source_rejected(self):
        row = asa_row(b"frozen bytes")
        with self.assertRaises(ValueError):
            fetch.validate_image({**row, "url": "https://example.com/"}, "development")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.png"
            target.write_bytes(b"frozen bytes")
            (root / "example.png").symlink_to(target)
            with self.assertRaises(ValueError):
                fetch.local_path(root, "example.png")


if __name__ == "__main__":
    unittest.main()
