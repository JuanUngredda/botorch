#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest import mock

import torch
from botorch.acquisition.analytic import PosteriorMean, ScalarizedPosteriorMean
from botorch.acquisition.cost_aware import GenericCostAwareUtility
from botorch.acquisition.knowledge_gradient import ContinuousKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement, qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective, ScalarizedObjective
from botorch.acquisition.utils import project_to_sample_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal

NO = "botorch.utils.testing.MockModel.num_outputs"


def mock_util(X, deltas):
    return 0.5 * deltas.sum(dim=0)


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
        continous_kg = ContinuousKnowledgeGradient(
            model,
            bounds=bounds,
            num_fantasies=NUM_FANTASIES,
            num_restarts=NUM_RESTARTS_INNER_OPTIMISER,
            raw_samples=NUM_RAW_SAMPLES_INNER_OPTIMISER,
        )
        X = torch.rand(4, 1, d, device=self.device, dtype=dtype)
        options = {"num_inner_restarts": 2, "raw_inner_samples": 3}
        val = continous_kg(X)

        # verify output shape
        self.assertEqual(val.size(), torch.Size([4]))
        # verify dtype
        self.assertEqual(val.dtype, dtype)
