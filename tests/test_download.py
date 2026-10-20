import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_dataset import SOURCES, download


class DownloadTests(unittest.TestCase):
    def manifest(self, payload):
        digest = hashlib.sha256(payload).hexdigest()
        return f"{digest}  S001/S001R04.edf\n".encode()

    def test_official_mirror_download_and_verified_cache(self):
        payload = b"synthetic EDF fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.download_dataset.urlopen", side_effect=[
                io.BytesIO(self.manifest(payload)), io.BytesIO(payload),
            ]) as request:
                paths = download([1], [4], root, source="aws")
                self.assertEqual(paths[0].read_bytes(), payload)
                self.assertEqual(request.call_args_list[1].args[0], SOURCES["aws"] + "/S001/S001R04.edf")
            with patch("scripts.download_dataset.urlopen", return_value=io.BytesIO(self.manifest(payload))) as request:
                download([1], [4], root, source="aws")
                self.assertEqual(request.call_count, 1)

    def test_corrupt_download_never_becomes_edf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.download_dataset.urlopen", side_effect=[
                io.BytesIO(self.manifest(b"correct")),
                io.BytesIO(b"corrupt"), io.BytesIO(b"corrupt"), io.BytesIO(b"corrupt"),
            ]):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    download([1], [4], root, source="aws")
            self.assertFalse((root / "S001/S001R04.edf").exists())
            self.assertFalse((root / "S001/S001R04.edf.part").exists())


if __name__ == "__main__":
    unittest.main()