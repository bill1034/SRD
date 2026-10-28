"""Numerical contracts for the Bayesian/spectral/convex reasoning modules."""
import importlib.util
import math
import pathlib
import unittest

import torch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "source" / "reasoning.py"
assert SOURCE.is_file(), "The genuine reasoning implementation has not been created yet"
SPEC = importlib.util.spec_from_file_location("reasoning_under_test", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class ReasoningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_student_t_full_density_matches_distribution_and_has_gradients(self):
        # Omitting the normalization/scale term or confusing scale and variance fails.
        z = torch.tensor([[0.2, -1.7]], dtype=torch.float64, requires_grad=True)
        mean = torch.tensor([0.4, -0.2], dtype=torch.float64)
        kappa = torch.tensor(3.0, dtype=torch.float64)
        alpha = torch.tensor(4.0, dtype=torch.float64)
        beta = torch.tensor([0.7, 1.9], dtype=torch.float64)
        scale = torch.sqrt(beta * (kappa + 1) / (alpha * kappa))
        want = torch.distributions.StudentT(8.0, mean, scale).log_prob(z)
        got = MOD.student_t_log_prob(z, mean, kappa, alpha, beta)
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
        got.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_nig_sufficient_statistics_have_hand_derived_posterior(self):
        # Prior (0,1,2,1), observations 1,3 gives (4/3,3,3,10/3).
        mean, kappa, alpha, beta = MOD.nig_posterior(
            torch.tensor([[1.0], [3.0]], dtype=torch.float64), 0.0, 1.0, 2.0, 1.0)
        torch.testing.assert_close(mean, torch.tensor([4 / 3], dtype=torch.float64))
        self.assertAlmostEqual(float(kappa), 3.0)
        self.assertAlmostEqual(float(alpha), 3.0)
        self.assertAlmostEqual(float(beta), 10 / 3)

    def test_bank_group_exclusion_empty_support_and_portable_restore(self):
        memory = MOD.BayesianMemory(2, 3, {"num_classes": 2, "capacity_enabled": False})
        keys = torch.tensor([[1.0, 2.0], [5.0, 8.0]])
        values = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        memory.set_bank(keys, values, torch.tensor([0, 1]), torch.tensor([4, 7]))
        query = torch.tensor([[1.1, 2.1]], requires_grad=True)
        residual, reliability, diag = memory(query, torch.tensor([7]))
        self.assertGreater(float(reliability[0]), 0)
        self.assertEqual(float(residual[0, 1]), 0.0)
        self.assertEqual(diag["retained_group_counts"].tolist(), [1])
        residual.sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        restored = MOD.BayesianMemory(2, 3, {"num_classes": 2, "capacity_enabled": False})
        restored.load_state_dict(memory.state_dict())
        actual = restored(query.detach(), torch.tensor([7]))[0]
        torch.testing.assert_close(actual, residual.detach())
        imported = MOD.BayesianMemory(2, 3, {"num_classes": 2, "capacity_enabled": False})
        imported.import_bank(memory.export_bank())
        torch.testing.assert_close(imported(query.detach())[0], memory(query.detach())[0])
        memory.set_bank(keys[:1], values[:1], torch.tensor([0]), torch.tensor([4]))
        residual, reliability, _ = memory(query, torch.tensor([4]))
        self.assertEqual(float(residual.abs().sum() + reliability.sum()), 0.0)

    def test_excluded_group_cannot_affect_standardization_or_posterior(self):
        config = {"num_classes": 1, "capacity_enabled": False}
        a, b = MOD.BayesianMemory(2, 2, config), MOD.BayesianMemory(2, 2, config)
        keys = torch.tensor([[1., 0.], [2., 1.], [3., 2.]])
        values = torch.eye(3)[:, :2]
        labels, groups = torch.zeros(3, dtype=torch.long), torch.tensor([0, 1, 2])
        a.set_bank(keys, values, labels, groups)
        changed = keys.clone(); changed[0] = 1e6
        b.set_bank(changed, values, labels, groups)
        query = torch.tensor([[2.2, 1.3]])
        torch.testing.assert_close(a(query, torch.tensor([0]))[0], b(query, torch.tensor([0]))[0])

    def test_capacity_is_feasible_and_real_read_is_not_renormalized(self):
        memory = MOD.BayesianMemory(2, 2, {
            "num_classes": 1, "capacity_enabled": True, "capacity_factor": 0.1,
            "dual_steps": 2, "null_cost": 10., "value_norm_clip": 1.})
        memory.set_bank(torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([[1., 0.], [1., 0.]]),
                        torch.tensor([0, 0]), torch.tensor([0, 1]))
        residual, reliability, diag = memory(torch.zeros(5, 2))
        self.assertLessEqual(float(diag["capacity_violation"]), 1e-6)
        self.assertLessEqual(float(reliability.mean()), 0.100001)
        torch.testing.assert_close(residual[:, 0], reliability)
        self.assertTrue(bool(diag["feasibility_repaired"]))

    def test_independent_group_representatives_are_not_counted_twice(self):
        config = {"num_classes": 1, "capacity_enabled": False}
        a, b = MOD.BayesianMemory(2, 2, config), MOD.BayesianMemory(2, 2, config)
        keys, values = torch.tensor([[0., 1.], [2., 3.]]), torch.eye(2)
        labels, groups = torch.tensor([0, 0]), torch.tensor([9, 10])
        a.set_bank(keys, values, labels, groups)
        b.set_bank(keys.repeat_interleave(3, 0), values.repeat_interleave(3, 0),
                   labels.repeat_interleave(3), groups.repeat_interleave(3))
        query = torch.tensor([[1., 1.]])
        torch.testing.assert_close(a(query)[0], b(query)[0])
        with self.assertRaises(ValueError):
            a.set_bank(keys, values, labels, groups, source_split="test")

    def test_spectral_budget_bypass_axial_periodicity_and_gradients(self):
        adapter = MOD.SpectralRoIAdapter(8, {"reduced_channels": 4, "max_rois": 2,
            "max_fraction": 1., "risk_threshold": -1., "layerscale_init": 0.1})
        features = torch.randn(5, 8, 7, 7, requires_grad=True)
        rois = torch.tensor([[0, 10, 10, 12, 4, 0.2]] * 5, dtype=torch.float32)
        probabilities = torch.tensor([[0.5, 0.5]] * 5)
        output, diag = adapter(features, rois, probabilities)
        self.assertEqual(int(diag["fft_rois"]), 2)
        unselected = ~diag["selected_mask"]
        torch.testing.assert_close(output[unselected], features[unselected], atol=0, rtol=0)
        rotated = rois.clone(); rotated[:, 5] += math.pi
        torch.testing.assert_close(adapter(features, rotated, probabilities)[0], output)
        output.square().mean().backward()
        self.assertGreater(float(adapter.reduce.weight.grad.abs().sum()), 0)
        self.assertGreater(float(adapter.frequency_response.grad.abs().sum()), 0)
        empty, _ = adapter(features[:0], rois[:0])
        self.assertEqual(tuple(empty.shape), (0, 8, 7, 7))

    def test_spectral_isotropic_input_remains_finite(self):
        adapter = MOD.SpectralRoIAdapter(8, {"reduced_channels": 4, "risk_threshold": -1.})
        output, diag = adapter(torch.ones(2, 8, 7, 7),
                              torch.tensor([[0., 0., 0., 10., 10., 0.]] * 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(diag["spectral_anisotropy"]).all())
        self.assertTrue((diag["feature_anisotropy"] < 0.05).all())
        self.assertTrue((diag["spectral_anisotropy"] < 0.05).all())

    def test_memory_capacity_is_independent_between_images(self):
        config = {"num_classes": 2, "capacity_factor": 0.6, "dual_steps": 3, "null_cost": 10.}
        memory = MOD.BayesianMemory(2, 2, config)
        memory.set_bank(torch.tensor([[0., 0.], [1., 1.], [8., 8.], [9., 9.]]),
                        torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),
                        torch.tensor([0, 0, 1, 1]), torch.arange(4))
        query = torch.tensor([[0., 0.], [1., 1.], [9., 9.]])
        combined = memory(query, query_image_ids=torch.tensor([0, 0, 1]))[0]
        separate = torch.cat((memory(query[:2])[0], memory(query[2:])[0]))
        torch.testing.assert_close(combined, separate)

    def test_spectral_budget_is_independent_between_images(self):
        adapter = MOD.SpectralRoIAdapter(8, {"max_rois": 1, "max_fraction": 1.,
            "risk_threshold": -1., "layerscale_init": 0.2})
        features = torch.randn(4, 8, 7, 7)
        rois = torch.tensor([[i // 2, 1., 1., 8. + i, 3., 0.] for i in range(4)])
        probability = torch.tensor([[0.5, 0.5], [0.99, 0.01], [0.95, 0.05], [0.8, 0.2]])
        together, diag = adapter(features, rois, probability)
        separate = torch.cat((adapter(features[:2], rois[:2], probability[:2])[0],
                              adapter(features[2:], rois[2:], probability[2:])[0]))
        self.assertEqual(int(diag["fft_rois"]), 2)
        torch.testing.assert_close(together, separate)

    def test_field_finite_step_result_is_independent_between_images(self):
        # A second dense component must not slow the first image's finite solve.
        field = MOD.SmoothTVReasoner({"neighbors": 3, "inner_steps": 3, "lambda_pairwise": 0.8})
        h0, residual, reliability = torch.randn(6, 3), torch.randn(6, 3), torch.rand(6)
        rois = torch.tensor([[i // 2 if i < 2 else 1, i, 0., 20., 10., 0.] for i in range(6)])
        keys = torch.ones(6, 2)
        together = field(h0, residual, reliability, rois, keys)[0]
        separate = torch.cat((field(h0[:2], residual[:2], reliability[:2], rois[:2], keys[:2])[0],
                              field(h0[2:], residual[2:], reliability[2:], rois[2:], keys[2:])[0]))
        torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-6)

    def test_field_gradient_and_hessian_match_independent_energy(self):
        config = {"unary_precision": 0.7, "eta_memory": 0.4, "lambda_pairwise": 0.3,
                  "rho_delta": 0.2, "position_scale": 2., "inner_steps": 1, "neighbors": 2}
        field = MOD.SmoothTVReasoner(config)
        h0 = torch.tensor([[0.1, 0.2], [0.5, -0.4], [-0.6, 0.3]])
        residual, reliability = torch.randn(3, 2), torch.tensor([0.2, 0.4, 0.8])
        rois = torch.tensor([[0., i, 0., 2., 2., 0.] for i in range(3)])
        got, diag = field(h0, residual, reliability, rois, torch.ones(3, 2))
        # Independent complete three-node graph and energy, in float64.
        center, target, c = h0.double(), (h0 + residual).double(), reliability.double()
        pairs = [(0, 1, math.exp(-1 / 32)), (1, 2, math.exp(-1 / 32)), (0, 2, math.exp(-1 / 8))]
        def energy(flat):
            u = flat.reshape(3, 2)
            out = 0.35 * (u - center).square().sum() + 0.2 * (c[:, None] * (u - target).square()).sum()
            for i, j, w in pairs:
                out = out + 0.3 * w * (torch.sqrt((u[i] - u[j]).square() + 0.04) - 0.2).sum()
            return out
        u = center.flatten().clone().requires_grad_()
        grad = torch.autograd.grad(energy(u), u)[0]
        expected = center - float(diag["step_size"]) * grad.reshape(3, 2)
        torch.testing.assert_close(got.double(), expected, atol=2e-7, rtol=2e-7)
        eigenvalues = torch.linalg.eigvalsh(torch.autograd.functional.hessian(energy, u))
        self.assertGreaterEqual(float(eigenvalues.min()), 0.7 - 1e-10)
        self.assertLessEqual(float(eigenvalues.max()), float(diag["lipschitz_bound"]) + 1e-6)

    def test_field_memory_gradient_matches_finite_difference(self):
        field = MOD.SmoothTVReasoner({"neighbors": 1, "inner_steps": 4})
        h0 = torch.tensor([[0.1, 0.2], [0.5, -0.4]])
        memory = torch.tensor([[0.3, -0.1], [0.2, 0.6]], requires_grad=True)
        reliability = torch.tensor([0.3, 0.8])
        rois, keys = torch.tensor([[0., i, 0., 2., 2., 0.] for i in range(2)]), torch.ones(2, 2)
        loss = field(h0, memory, reliability, rois, keys)[0].square().sum()
        analytic = torch.autograd.grad(loss, memory)[0][0, 1]
        plus, minus = memory.detach().clone(), memory.detach().clone()
        plus[0, 1] += 1e-3; minus[0, 1] -= 1e-3
        finite = (field(h0, plus, reliability, rois, keys)[0].square().sum()
                  - field(h0, minus, reliability, rois, keys)[0].square().sum()) / 2e-3
        torch.testing.assert_close(analytic, finite, atol=2e-5, rtol=5e-3)

    def test_field_single_node_is_memory_tether_not_double_injection(self):
        # With a=2, eta*c=2, optimum is h0 + memory/2, not h0 + memory.
        field = MOD.SmoothTVReasoner({"unary_precision": 2., "eta_memory": 4., "inner_steps": 1})
        h0 = torch.tensor([[2., -3.]], requires_grad=True)
        residual = torch.tensor([[4., -2.]], requires_grad=True)
        output, diag = field(h0, residual, torch.tensor([0.5]),
                             torch.tensor([[0., 1., 1., 3., 2., 0.]]), torch.ones(1, 2))
        torch.testing.assert_close(output, torch.tensor([[4., -4.]]))
        output.sum().backward()
        torch.testing.assert_close(residual.grad, torch.tensor([[0.5, 0.5]]))
        self.assertEqual(int(diag["edge_count"]), 0)

    def test_field_energy_decreases_no_cross_image_edges_and_permutation(self):
        field = MOD.SmoothTVReasoner({"neighbors": 2, "inner_steps": 20, "lambda_pairwise": 0.4})
        h0 = torch.randn(6, 4, requires_grad=True)
        residual = torch.randn(6, 4, requires_grad=True)
        rois = torch.tensor([[i // 3, i % 3, 0, 2, 1, 0] for i in range(6)], dtype=torch.float32)
        keys, reliability = torch.randn(6, 3), torch.rand(6)
        out, diag = field(h0, residual, reliability, rois, keys)
        energies = diag["energy_history"]
        self.assertTrue(torch.all(energies[1:] <= energies[:-1] + 2e-5))
        edges = diag["edges"]
        self.assertTrue(torch.equal(rois[edges[:, 0], 0], rois[edges[:, 1], 0]))
        p = torch.tensor([3, 0, 5, 2, 1, 4])
        other, _ = field(h0[p], residual[p], reliability[p], rois[p], keys[p])
        torch.testing.assert_close(other, out[p], atol=2e-6, rtol=2e-6)
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(h0.grad).all() and torch.isfinite(residual.grad).all())
        self.assertGreater(float(residual.grad.abs().sum()), 0)
        empty, _ = field(h0[:0], residual[:0], reliability[:0], rois[:0], keys[:0])
        self.assertEqual(tuple(empty.shape), (0, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "A CUDA device is optional")
    def test_cuda_full_size_reasoning_chain_backward_and_bank_immutability(self):
        adapter = MOD.SpectralRoIAdapter(256, {"max_rois": 8, "risk_threshold": -1.}).cuda()
        memory = MOD.BayesianMemory(32, 192, {"num_classes": 2, "dual_steps": 8}).cuda()
        field = MOD.SmoothTVReasoner({"inner_steps": 8}).cuda()
        memory.set_bank(torch.randn(12, 32), torch.randn(12, 192),
                        torch.arange(12) % 2, torch.arange(12))
        before = memory.export_bank()
        feats = torch.randn(16, 256, 7, 7, device="cuda", requires_grad=True)
        rois = torch.tensor([[i // 8, i % 8, 1., 8., 3., 0.1] for i in range(16)], device="cuda")
        keys = torch.randn(16, 32, device="cuda", requires_grad=True)
        h0 = torch.randn(16, 192, device="cuda", requires_grad=True)
        corrected, _ = adapter(feats, rois, torch.softmax(torch.randn(16, 3, device="cuda"), 1))
        residual, reliability, _ = memory(keys, query_image_ids=rois[:, 0].long())
        output, diag = field(h0, residual, reliability, rois, keys)
        (corrected.square().mean() + output.square().mean()).backward()
        for gradient in (feats.grad, keys.grad, h0.grad, adapter.frequency_response.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertTrue(torch.all(diag["energy_history"][1:] <= diag["energy_history"][:-1] + 2e-4))
        after = memory.export_bank()
        for name in ("keys", "values", "labels", "groups"):
            torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
