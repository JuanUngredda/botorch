#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from botorch.acquisition.knowledge_gradient import MCKnowledgeGradient
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class TestDicreteKnowledgeGradient(BotorchTestCase):
    def test_evaluate_kg(self):
        # a thorough test using real model and dtype double
        d = 2
        NUM_FANTASIES = 3
        NUM_RESTARTS_INNER_OPTIMISER = 1
        NUM_RAW_SAMPLES_INNER_OPTIMISER = 20
        dtype = torch.double
        bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype).repeat(1, d)
        train_X = torch.rand(3, d, device=self.device, dtype=dtype)
        train_Y = torch.rand(3, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        continous_kg = MCKnowledgeGradient(
            model,
            bounds=bounds,
            num_fantasies=NUM_FANTASIES,
            num_restarts=NUM_RESTARTS_INNER_OPTIMISER,
            raw_samples=NUM_RAW_SAMPLES_INNER_OPTIMISER,
        )
        X = torch.rand(4, 1, d, device=self.device, dtype=dtype)
        val = continous_kg(X)

        # verify output shape
        self.assertEqual(val.size(), torch.Size([4]))
        # verify dtype
        self.assertEqual(val.dtype, dtype)
