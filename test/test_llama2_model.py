import unittest

import torch
from model.llama2 import LLAMA2
from test.test_config import LLAMA2_CONFIG_test


class TestLlama2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = LLAMA2(LLAMA2_CONFIG_test)

    def test_llama2(self):
        print(self.model)
