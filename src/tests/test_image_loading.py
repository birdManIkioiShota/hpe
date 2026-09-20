from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from hpe.data.dataset import read_rgb_image


class ImageReadTests(unittest.TestCase):
    def test_retry_decode_error_preserves_pixels_and_records_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.png"
            pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            Image.fromarray(pixels).save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            original = Image.Image.convert
            calls = 0

            def flaky_convert(image, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("broken data stream when reading image file")
                return original(image, *args, **kwargs)

            with patch.object(Image.Image, "convert", flaky_convert), patch("hpe.data.dataset.time.sleep"):
                decoded = read_rgb_image(path, expected_sha256=digest, error_dir=root / "errors")
            np.testing.assert_array_equal(np.asarray(decoded), pixels)
            records = [json.loads(line) for log in (root / "errors").glob("*.jsonl")
                       for line in log.read_text().splitlines()]
            self.assertEqual([item["event"] for item in records],
                             ["image_read_failed", "image_read_recovered"])
            self.assertEqual(records[0]["observed_sha256"], digest)
            self.assertEqual(records[0]["image_path"], str(path))

    def test_persistent_decode_or_hash_error_stops_after_three_attempts(self):
        for failure in ("decode", "hash"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "image.png"
                if failure == "decode":
                    path.write_bytes(b"invalid png stream")
                    expected = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    Image.new("RGB", (4, 4)).save(path)
                    expected = "0" * 64
                with patch("hpe.data.dataset.time.sleep"), self.assertRaises(OSError) as raised:
                    read_rgb_image(path, expected_sha256=expected, error_dir=root / "errors")
                self.assertIn(str(path), str(raised.exception))
                self.assertIn("after 3 reads", str(raised.exception))
                records = [json.loads(line) for log in (root / "errors").glob("*.jsonl")
                           for line in log.read_text().splitlines()]
                self.assertEqual(len(records), 3)
                self.assertTrue(all(item["event"] == "image_read_failed" for item in records))


if __name__ == "__main__":
    unittest.main()
