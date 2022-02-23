#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import DiscreteKnowledgeGradient
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class TestDicreteKnowledgeGradient(BotorchTestCase):
    def test_evaluate_kg(self):
        # a thorough test using real model and dtype double
        d = 2
        NUM_DISCRETE_X = 100
        dtype = torch.double
        bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype).repeat(1, d)
        train_X = torch.rand(3, d, device=self.device, dtype=dtype)
        train_Y = torch.rand(3, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        dKG = DiscreteKnowledgeGradient(
            model=model,
            bounds=bounds,
            num_discrete_points=NUM_DISCRETE_X,
            X_discretisation=None,
        )
        X = torch.rand(4, 1, d, device=self.device, dtype=dtype)
        val = dKG(X)

        # verify output shape
        self.assertEqual(val.size(), torch.Size([4]))
        # verify dtype
        self.assertEqual(val.dtype, dtype)
