import os
import tempfile
import unittest

import pandas as pd

from utils import ml_common as mc
from utils.model_train import build_dataset



class BuildDatasetTest(unittest.TestCase):

    def test_build_dataset(self):
        again = build_dataset("2024-02-01")
        pd.testing.assert_series_equal(self.data["split"], again["split"])


if __name__ == "__main__":
    unittest.main()
