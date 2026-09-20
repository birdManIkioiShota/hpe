from __future__ import annotations

import unittest

from hpe.datasets.single_pose import landmark_crop_xyxy


class LandmarkCropTests(unittest.TestCase):
    def test_official_6drepnet_margin_order(self):
        crop = landmark_crop_xyxy([[10.0, 110.0], [20.0, 220.0]])
        self.assertEqual(crop, [-30.0, -60.0, 166.0, 253.6])


if __name__ == "__main__":
    unittest.main()
