"""Conditional spectral evidence, frozen-space Bayesian memory and convex graph inference.

All modules are self-contained PyTorch code. Bayesian claims apply to the fixed key
space only; the graph guarantee concerns its fixed-coefficient evidence energy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def nig_posterior(observations, prior_mean=0., kappa0=1., alpha0=2., beta0=1.):
    """Conjugate diagonal Normal-Inverse-Gamma update from independent observations."""
    n = observations.shape[0]
    base = observations.new_tensor(prior_mean)
    kappa = observations.new_tensor(kappa0 + n)
    alpha = observations.new_tensor(alpha0 + n / 2.)
    if n == 0:
        shape = observations.shape[1:]
        return base.expand(shape), kappa, alpha, observations.new_full(shape, beta0)
    average = observations.mean(0)
    scatter = (observations - average).square().sum(0)
    mean = (kappa0 * base + n * average) / kappa
    beta = beta0 + 0.5 * scatter + kappa0 * n * (average - base).square() / (2. * kappa)
    return mean, kappa, alpha, beta


def student_t_log_prob(z, mean, kappa, alpha, beta):
    """Full predictive log density, including scale and normalization terms."""
    nu = 2. * alpha
    scale_sq = beta * (kappa + 1.) / (alpha * kappa)
    return (torch.lgamma((nu + 1.) / 2.) - torch.lgamma(nu / 2.)
            - 0.5 * (torch.log(nu * math.pi) + torch.log(scale_sq))
            - ((nu + 1.) / 2.) * torch.log1p((z - mean).square() / (nu * scale_sq)))


class SpectralRoIAdapter(nn.Module):
    """Vectorized FFT residual on a bounded subset of already aligned RoIs.

    Geometry uses [batch,cx,cy,w,h,theta]. Directions are axial and expressed in
    the canonical RoI frame. This adapter does not claim continuous equivariance.
    The structure tensor and the spectrum are different observed feature cues,
    not regression uncertainty or independently sampled Bayesian evidence.
    """
    DEFAULTS = {
        "reduced_channels": 64, "max_rois": 128, "max_fraction": 0.5,
        "risk_threshold": 0.25, "small_object_scale": 64.,
        "entropy_weight": 0.45, "size_weight": 0.3, "direction_weight": 0.25,
        "spectral_conflict_weight": 0.5, "anisotropy_floor": 0.05,
        "gate_slope": 4., "layerscale_init": 1e-3,
        "frequency_init_std": 0.02, "numerical_eps": 1e-6,
    }

    def __init__(self, channels, cfg=None):
        super().__init__()
        self.cfg = dict(self.DEFAULTS, **(cfg or {}))
        reduced = int(self.cfg["reduced_channels"])
        if channels < 1 or reduced < 1 or int(self.cfg["max_rois"]) < 0:
            raise ValueError("Channels must be positive and max_rois nonnegative")
        if not 0 <= self.cfg["max_fraction"] <= 1:
            raise ValueError("max_fraction must lie in [0,1]")
        self.reduce = nn.Conv2d(channels, reduced, 1, bias=False)
        self.expand = nn.Conv2d(reduced, channels, 1, bias=False)
        self.frequency_response = nn.Parameter(torch.empty(reduced, 3))
        nn.init.normal_(self.frequency_response, std=self.cfg["frequency_init_std"])
        self.layerscale = nn.Parameter(torch.full((1, channels, 1, 1), self.cfg["layerscale_init"]))

    def _cheap_risk(self, feats, rois, class_prob, orientation_hint):
        eps = self.cfg["numerical_eps"]
        # Channel RMS avoids cancellation of signed appearance features.
        signal = (feats.detach().float().square().mean(1) + eps).sqrt()
        dx = signal[:, 1:-1, 2:] - signal[:, 1:-1, :-2]
        dy = signal[:, 2:, 1:-1] - signal[:, :-2, 1:-1]
        xx, yy, xy = dx.square().mean((1, 2)), dy.square().mean((1, 2)), (dx * dy).mean((1, 2))
        norm = ((xx - yy).square() + 4. * xy.square() + eps ** 2).sqrt() - eps
        anisotropy = (norm / (xx + yy + eps)).clamp(0., 1.)
        # The spatial long axis is perpendicular to the dominant gradient axis.
        cos2 = -(xx - yy) / (norm + eps)
        if orientation_hint is not None:
            hint = orientation_hint.detach().float()
            if hint.ndim == 1:
                cos2 = torch.cos(2. * hint)
            elif hint.shape[1] == 2:
                magnitude = hint.norm(dim=1).clamp_min(eps)
                cos2 = hint[:, 0] / magnitude
                anisotropy = magnitude.clamp(0., 1.)
            else:
                # Optional orientation energies are already in the local frame.
                probabilities = hint.clamp_min(0.)
                probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(eps)
                angle = torch.arange(hint.shape[1], device=hint.device) * (2. * math.pi / hint.shape[1])
                axial_x = (probabilities * torch.cos(2. * angle)).sum(1)
                axial_y = (probabilities * torch.sin(2. * angle)).sum(1)
                anisotropy = (axial_x.square() + axial_y.square()).sqrt().clamp(0., 1.)
                cos2 = axial_x / anisotropy.clamp_min(eps)
        aspect = (rois[:, 3] - rois[:, 4]).abs() / (rois[:, 3] + rois[:, 4]).clamp_min(eps)
        directional_conflict = 0.5 * (1. - cos2.clamp(-1., 1.)) * aspect * anisotropy
        directional_conflict = directional_conflict * (anisotropy >= self.cfg["anisotropy_floor"])
        if class_prob is None:
            entropy = signal.new_zeros(signal.shape[0])
        else:
            p = class_prob.detach().float().clamp_min(eps)
            if p.ndim != 2 or p.shape[0] != len(feats) or p.shape[1] < 2:
                raise ValueError("class_prob must include foreground AND background classes")
            p = p / p.sum(1, keepdim=True)
            entropy = -(p * p.log()).sum(1) / math.log(p.shape[1])
        size = (self.cfg["small_object_scale"] / (rois[:, 3] * rois[:, 4]).clamp_min(eps).sqrt()).clamp(max=1.)
        risk = (self.cfg["entropy_weight"] * entropy + self.cfg["size_weight"] * size
                + self.cfg["direction_weight"] * directional_conflict)
        return risk, anisotropy

    def forward(self, feats, rois, class_prob=None, orientation_hint=None):
        if feats.ndim != 4 or feats.shape[2] < 3 or feats.shape[3] < 3:
            raise ValueError("Expected [N,C,H,W] aligned RoIs with H,W >= 3")
        if rois.shape != (feats.shape[0], 6):
            raise ValueError("RoIs must have shape [N,6]")
        n = feats.shape[0]
        selected = torch.zeros(n, device=feats.device, dtype=torch.bool)
        empty = feats.new_zeros(n, dtype=torch.float32)
        diag = {"selected_mask": selected, "selected_indices": selected.nonzero().flatten(),
                "fft_rois": 0, "cheap_risk": empty, "feature_anisotropy": empty,
                "spectral_anisotropy": empty, "spectral_conflict": empty,
                "direction_source": "canonical_roi_structure_tensor"}
        if n == 0:
            return feats, diag
        risk, feature_anisotropy = self._cheap_risk(feats, rois.float(), class_prob, orientation_hint)
        # Budgets are per image: unrelated batch members cannot alter selection.
        for image in torch.unique(rois[:, 0]):
            image_mask = rois[:, 0] == image
            count = int(image_mask.sum())
            budget = min(int(self.cfg["max_rois"]), int(math.ceil(count * self.cfg["max_fraction"])))
            candidates = torch.nonzero(image_mask & (risk >= self.cfg["risk_threshold"])).flatten()
            if budget > 0 and candidates.numel():
                chosen = candidates[torch.topk(risk[candidates], min(budget, len(candidates)), sorted=False).indices]
                selected[chosen] = True
        indices = torch.nonzero(selected).flatten()
        diag.update(selected_indices=indices, fft_rois=len(indices), cheap_risk=risk.detach(),
                    feature_anisotropy=feature_anisotropy.detach())
        if not len(indices):
            return feats, diag
        reduced = self.reduce(feats[indices]).float()
        height, width = reduced.shape[-2:]
        spectrum = torch.fft.rfft2(reduced, norm="ortho")
        fy = torch.fft.fftfreq(height, device=feats.device).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=feats.device).view(1, width // 2 + 1)
        radius2 = fx.square() + fy.square()
        eps = self.cfg["numerical_eps"]
        basis = torch.stack((radius2 / radius2.max().clamp_min(eps),
                             (fx.square() - fy.square()) / radius2.clamp_min(eps),
                             2. * fx * fy / radius2.clamp_min(eps)))
        # An even real frequency response preserves the real-valued FFT contract.
        response = torch.tanh(torch.einsum("ck,khw->chw", self.frequency_response.float(), basis))
        response = response * (radius2 > 0.)  # DC remains on the unchanged main path.
        correction = torch.fft.irfft2(spectrum * response[None], s=(height, width), norm="ortho")
        # rFFT interior columns represent both positive and negative frequencies.
        multiplicity = torch.full_like(fx, 2.)
        multiplicity[:, 0] = 1.
        if width % 2 == 0:
            multiplicity[:, -1] = 1.
        power = spectrum.abs().square().mean(1) * multiplicity
        xx, yy, xy = (power * fx.square()).sum((1, 2)), (power * fy.square()).sum((1, 2)), (power * fx * fy).sum((1, 2))
        norm = ((xx - yy).square() + 4. * xy.square() + eps ** 2).sqrt() - eps
        anisotropy = (norm / (xx + yy + eps)).clamp(0., 1.)
        spatial_cos2 = -(xx - yy) / (norm + eps)
        aspect = (rois[indices, 3] - rois[indices, 4]).abs() / (rois[indices, 3] + rois[indices, 4]).clamp_min(eps)
        conflict = 0.5 * (1. - spatial_cos2.clamp(-1., 1.)) * anisotropy * aspect
        conflict = conflict * (anisotropy >= self.cfg["anisotropy_floor"])
        gate = torch.sigmoid(self.cfg["gate_slope"] * (risk[indices] - self.cfg["risk_threshold"])
                             + self.cfg["spectral_conflict_weight"] * conflict)
        residual = self.expand(correction.to(feats.dtype)) * self.layerscale * gate[:, None, None, None]
        output = feats.clone()
        output[indices] = feats[indices] + residual.to(feats.dtype)
        diag["spectral_anisotropy"] = empty.clone().index_copy(0, indices, anisotropy.detach())
        diag["spectral_conflict"] = empty.clone().index_copy(0, indices, conflict.detach())
        return output, diag


class BayesianMemory(nn.Module):
    """Train-only, independent-group NIG prototypes with null/capacity reading.

    A set_bank call freezes detached representatives. Forward never updates them.
    Training query groups are excluded BEFORE standardization and posterior fit.
    An unconverged finite dual solve is reported; any excess column mass is
    conservatively transferred to null, producing a feasible approximate read.
    """
    DEFAULTS = {
        "num_classes": 6, "prior_mean": 0., "kappa0": 1., "alpha0": 2., "beta0": 1.,
        "standardization_floor": 0.05, "epsilon": 0.3, "null_cost": 2.,
        "null_prior": 0.25, "capacity_enabled": True, "capacity_factor": 1.25,
        "dual_steps": 80, "solver_tolerance": 1e-4, "value_norm_clip": 1.,
        "numerical_eps": 1e-8,
    }

    def __init__(self, key_dim=32, value_dim=192, cfg=None):
        super().__init__()
        self.cfg = dict(self.DEFAULTS, **(cfg or {}))
        self.key_dim, self.value_dim = int(key_dim), int(value_dim)
        if min(self.key_dim, self.value_dim, self.cfg["num_classes"]) < 1:
            raise ValueError("Dimensions and class count must be positive")
        if min(self.cfg[k] for k in ("kappa0", "alpha0", "beta0", "epsilon", "standardization_floor", "value_norm_clip")) <= 0:
            raise ValueError("NIG, temperature, standardization and norm parameters must be positive")
        if not 0 < self.cfg["null_prior"] < 1 or self.cfg["capacity_factor"] < 0:
            raise ValueError("Invalid null prior or capacity factor")
        self.register_buffer("bank_keys", torch.empty(0, self.key_dim))
        self.register_buffer("bank_values", torch.empty(0, self.value_dim))
        self.register_buffer("bank_labels", torch.empty(0, dtype=torch.long))
        self.register_buffer("bank_groups", torch.empty(0, dtype=torch.long))

    def set_bank(self, keys, values, labels, groups, source_split="train"):
        if source_split != "train":
            raise ValueError("The reference bank accepts training representatives only")
        if keys.ndim != 2 or keys.shape[1] != self.key_dim or values.shape != (len(keys), self.value_dim):
            raise ValueError("Bank key/value dimensions do not match")
        if labels.shape != (len(keys),) or groups.shape != (len(keys),):
            raise ValueError("Bank labels and groups must be one-dimensional")
        if labels.is_floating_point() or groups.is_floating_point():
            raise ValueError("Labels/groups must use integer identifiers")
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError("Bank features must be finite")
        if len(labels) and (labels.min() < 0 or labels.max() >= self.cfg["num_classes"]):
            raise ValueError("Bank labels fall outside configured foreground ontology")
        device = self.bank_keys.device
        keys, values = keys.detach().float().to(device), values.detach().float().to(device)
        labels, groups = labels.detach().long().to(device), groups.detach().long().to(device)
        if len(keys):
            identities, inverse = torch.unique(torch.stack((labels, groups), 1), dim=0, return_inverse=True)
            count = torch.bincount(inverse, minlength=len(identities)).to(keys.dtype)[:, None]
            aggregate_keys = torch.zeros(len(identities), self.key_dim, device=device).index_add(0, inverse, keys) / count
            aggregate_values = torch.zeros(len(identities), self.value_dim, device=device).index_add(0, inverse, values) / count
            labels, groups = identities[:, 0], identities[:, 1]
            keys, values = aggregate_keys, aggregate_values
        self.bank_keys, self.bank_values = keys, values
        self.bank_labels, self.bank_groups = labels, groups

    def export_bank(self):
        return {"schema_version": 1, "source_split": "train", "key_dim": self.key_dim,
                "value_dim": self.value_dim, "num_classes": self.cfg["num_classes"],
                "keys": self.bank_keys.detach().cpu().clone(), "values": self.bank_values.detach().cpu().clone(),
                "labels": self.bank_labels.detach().cpu().clone(), "groups": self.bank_groups.detach().cpu().clone()}

    def import_bank(self, payload):
        if payload.get("schema_version") != 1 or payload.get("source_split") != "train":
            raise ValueError("Unrecognized or nontraining bank")
        if (payload.get("key_dim"), payload.get("value_dim"), payload.get("num_classes")) != (self.key_dim, self.value_dim, self.cfg["num_classes"]):
            raise ValueError("Bank dimensions or class ontology differ")
        self.set_bank(payload["keys"], payload["values"], payload["labels"], payload["groups"])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Dynamic banks must resize before Module's ordinary checked copy.
        for name in ("bank_keys", "bank_values", "bank_labels", "bank_groups"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key], device=self.bank_keys.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _read_pool(self, queries, keep):
        refs, vals, labels = self.bank_keys[keep], self.bank_values[keep], self.bank_labels[keep]
        n, device = len(queries), queries.device
        if not len(refs):
            zeros = queries.new_zeros(n)
            return queries.new_zeros(n, self.value_dim), zeros, {
                "null_mass": torch.ones_like(zeros), "capacity_violation": queries.new_zeros(()),
                "raw_capacity_violation": queries.new_zeros(()), "dual_residual": queries.new_zeros(()),
                "feasibility_repaired": False, "converged": True, "row_mass_error": queries.new_zeros(())}
        mean = refs.mean(0)
        std = refs.std(0, unbiased=False).clamp_min(self.cfg["standardization_floor"])
        standardized_refs, standardized_query = (refs - mean) / std, (queries - mean) / std
        present = torch.unique(labels, sorted=True)
        costs, values, counts = [], [], []
        for label in present:
            selected = labels == label
            posterior = nig_posterior(standardized_refs[selected], self.cfg["prior_mean"],
                                      self.cfg["kappa0"], self.cfg["alpha0"], self.cfg["beta0"])
            costs.append(-student_t_log_prob(standardized_query, *posterior).mean(1))
            value = vals[selected].mean(0)
            values.append(value * (self.cfg["value_norm_clip"] / value.norm().clamp_min(self.cfg["numerical_eps"])).clamp(max=1.))
            counts.append(selected.sum())
        cost = torch.stack(costs, 1)
        value = torch.stack(values)
        count = torch.stack(counts).float()
        class_prior = count / count.sum()
        prior = torch.cat((cost.new_tensor([self.cfg["null_prior"]]), (1. - self.cfg["null_prior"]) * class_prior))
        cost = torch.cat((cost.new_full((n, 1), self.cfg["null_cost"]), cost), 1)
        capacity = self.cfg["capacity_factor"] * class_prior
        dual = cost.new_zeros(len(present))
        epsilon = self.cfg["epsilon"]
        steps = int(self.cfg["dual_steps"]) if self.cfg["capacity_enabled"] else 0
        # Row masses sum to one; epsilon is a conservative dual step size.
        for _ in range(steps):
            adjusted = cost + torch.cat((dual.new_zeros(1), dual))[None]
            probabilities = torch.softmax(prior.log()[None] - adjusted / epsilon, dim=1)
            dual = torch.relu(dual + epsilon * (probabilities[:, 1:].mean(0) - capacity))
        adjusted = cost + torch.cat((dual.new_zeros(1), dual))[None]
        probabilities = torch.softmax(prior.log()[None] - adjusted / epsilon, dim=1)
        real = probabilities[:, 1:]
        if self.cfg["capacity_enabled"]:
            mass = real.mean(0)
            raw_violation = (mass - capacity).clamp_min(0.).max()
            residual = (dual - torch.relu(dual + epsilon * (mass - capacity))).abs().max() / epsilon
            # Conservative feasible fallback preserves row mass by sending excess to null.
            shrink = (capacity / mass.clamp_min(self.cfg["numerical_eps"])).clamp(max=1.)
            real = real * shrink[None]
            violation = (real.mean(0) - capacity).clamp_min(0.).max()
            repaired = bool((raw_violation > self.cfg["solver_tolerance"]).detach())
            converged = bool((residual <= self.cfg["solver_tolerance"]).detach())
        else:
            raw_violation = residual = violation = cost.new_zeros(())
            repaired, converged = False, True
        reliability = real.sum(1)
        null = 1. - reliability
        result = real @ value  # Do not renormalize the real columns.
        diag = {"null_mass": null.detach(), "capacity_violation": violation.detach(),
                "raw_capacity_violation": raw_violation.detach(), "dual_residual": residual.detach(),
                "feasibility_repaired": repaired, "converged": converged,
                "row_mass_error": (real.sum(1) + null - 1.).abs().max().detach()}
        return result, reliability, diag

    def forward(self, query_keys, query_groups=None, query_image_ids=None):
        if query_keys.ndim != 2 or query_keys.shape[1] != self.key_dim:
            raise ValueError("Unexpected query key dimension")
        if query_keys.device != self.bank_keys.device:
            raise ValueError("Move BayesianMemory to the query device before reading")
        if not torch.isfinite(query_keys).all():
            raise ValueError("Query keys must be finite")
        n = len(query_keys)
        if query_groups is not None and (query_groups.shape != (n,) or query_groups.is_floating_point()):
            raise ValueError("query_groups must be integer identifiers of shape [N]")
        if query_image_ids is not None and (query_image_ids.shape != (n,) or query_image_ids.is_floating_point()):
            raise ValueError("query_image_ids must be integer identifiers of shape [N]")
        query_keys = query_keys.float()
        result = query_keys.new_zeros(n, self.value_dim)
        reliability = query_keys.new_zeros(n)
        null = query_keys.new_ones(n)
        retained = torch.zeros(n, device=query_keys.device, dtype=torch.long)
        maximums = {name: query_keys.new_zeros(()) for name in (
            "capacity_violation", "raw_capacity_violation", "dual_residual", "row_mass_error")}
        all_converged, any_repaired = True, False
        image_ids = torch.zeros(n, device=query_keys.device, dtype=torch.long) if query_image_ids is None else query_image_ids.to(query_keys.device)
        group_ids = torch.zeros(n, device=query_keys.device, dtype=torch.long) if query_groups is None else query_groups.to(query_keys.device)
        partitions = torch.unique(torch.stack((image_ids, group_ids), 1), dim=0)
        for image, group in partitions:
            select = (image_ids == image) & (group_ids == group)
            if not select.any():
                continue
            keep = torch.ones(len(self.bank_keys), device=query_keys.device, dtype=torch.bool) if query_groups is None else self.bank_groups != group
            read, strength, diag = self._read_pool(query_keys[select], keep)
            result[select], reliability[select], null[select] = read, strength, diag["null_mass"]
            retained[select] = torch.unique(self.bank_groups[keep]).numel()
            for name in maximums:
                maximums[name] = torch.maximum(maximums[name], diag[name])
            all_converged = all_converged and diag["converged"]
            any_repaired = any_repaired or diag["feasibility_repaired"]
        return result, reliability, dict(maximums, null_mass=null, retained_group_counts=retained,
            converged=all_converged, feasibility_repaired=any_repaired,
            posterior_scope="fixed_key_diagonal_NIG", reference_count=len(self.bank_keys))


class SmoothTVReasoner(nn.Module):
    """Strongly convex smooth-TV refinement in unbounded RoI evidence space.

    The memory residual enters ONLY the soft target h0 + residual. Weights and
    graph are formed once per call and remain fixed throughout the inner solve.
    No operation smooths RoI coordinates or merges candidate identities.
    """
    DEFAULTS = {"unary_precision": 1., "mu_floor": 0.2, "lambda_pairwise": 0.05,
                "eta_memory": 0.1, "rho_delta": 0.1, "inner_steps": 5,
                "neighbors": 8, "position_scale": 3., "feature_temperature": 0.5,
                "weight_max": 1., "numerical_eps": 1e-8}

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = dict(self.DEFAULTS, **(cfg or {}))
        if self.cfg["unary_precision"] < self.cfg["mu_floor"] or self.cfg["mu_floor"] <= 0:
            raise ValueError("The unary precision must preserve a positive strong-convexity floor")
        if min(self.cfg[k] for k in ("rho_delta", "position_scale", "feature_temperature")) <= 0:
            raise ValueError("Smoothing/scale parameters must be positive")
        if min(self.cfg[k] for k in ("lambda_pairwise", "eta_memory", "weight_max", "inner_steps", "neighbors")) < 0:
            raise ValueError("Graph coefficients and budgets must be nonnegative")

    def _graph(self, rois, keys):
        edges = []
        n = len(rois)
        neighbors = int(self.cfg["neighbors"])
        if n < 2 or neighbors == 0:
            return torch.empty(0, 2, device=rois.device, dtype=torch.long), keys.new_empty(0)
        # Geometry controls connectivity; it is never an optimization variable.
        positions = rois[:, 1:3].detach().float()
        sizes = rois[:, 3:5].detach().float().clamp_min(self.cfg["numerical_eps"]).prod(1).sqrt()
        normalized_keys = F.normalize(keys.detach().float(), dim=1)
        for image in torch.unique(rois[:, 0]):
            nodes = torch.nonzero(rois[:, 0] == image).flatten()
            if len(nodes) < 2:
                continue
            distance = torch.cdist(positions[nodes], positions[nodes])
            distance = distance / ((sizes[nodes, None] + sizes[None, nodes]) * 0.5).clamp_min(self.cfg["numerical_eps"])
            distance.fill_diagonal_(float("inf"))
            nearest = torch.topk(distance, min(neighbors, len(nodes) - 1), largest=False).indices
            source = nodes[:, None].expand_as(nearest).reshape(-1)
            destination = nodes[nearest].reshape(-1)
            pairs = torch.stack((torch.minimum(source, destination), torch.maximum(source, destination)), 1)
            edges.append(torch.unique(pairs, dim=0))
        if not edges:
            return torch.empty(0, 2, device=rois.device, dtype=torch.long), keys.new_empty(0)
        edge = torch.cat(edges, 0)
        left, right = edge[:, 0], edge[:, 1]
        scale = 0.5 * (sizes[left] + sizes[right])
        distance2 = ((positions[left] - positions[right]) / scale[:, None]).square().sum(1)
        cosine_distance = (1. - (normalized_keys[left] * normalized_keys[right]).sum(1)).clamp(0., 2.)
        weights = self.cfg["weight_max"] * torch.exp(
            -distance2 / (2. * self.cfg["position_scale"] ** 2)
            -cosine_distance / self.cfg["feature_temperature"])
        return edge, weights

    def forward(self, h0, memory_residual, reliability, rois, keys):
        n = len(h0)
        if h0.ndim != 2 or memory_residual.shape != h0.shape or reliability.shape != (n,):
            raise ValueError("Evidence, residual and reliability dimensions differ")
        if rois.shape != (n, 6) or keys.ndim != 2 or len(keys) != n:
            raise ValueError("RoI/key dimensions do not match evidence")
        if not all(torch.isfinite(t).all() for t in (h0, memory_residual, reliability, rois, keys)):
            raise ValueError("Graph inputs must be finite")
        original_dtype = h0.dtype
        h0, memory_residual = h0.float(), memory_residual.float()
        if not n:
            return h0.to(original_dtype), {"energy_history": h0.new_zeros(1), "edge_count": 0,
                "edges": torch.empty(0, 2, device=h0.device, dtype=torch.long),
                "step_size": h0.new_tensor(0.), "lipschitz_bound": h0.new_tensor(self.cfg["unary_precision"]),
                "gradient_residual": h0.new_tensor(0.), "coefficients_fixed": True}
        # Fixed during this solve; gradients can still flow through the memory tether.
        c = reliability.float().clamp(0., 1.)
        target = h0 + memory_residual
        edges, weights = self._graph(rois, keys)
        left, right = edges[:, 0], edges[:, 1]
        degree = h0.new_zeros(n)
        if len(edges):
            degree = degree.index_add(0, left, weights).index_add(0, right, weights)
        a, eta, lam, delta = (self.cfg[k] for k in ("unary_precision", "eta_memory", "lambda_pairwise", "rho_delta"))
        # The graph is block diagonal by image. Certify each block separately,
        # otherwise a dense unrelated image changes a finite-step prediction.
        image_ids = torch.unique(rois[:, 0])
        node_steps = torch.zeros_like(c)
        image_bounds = []
        for image in image_ids:
            members = rois[:, 0] == image
            image_bound = (a + eta * c[members]).max() + 2. * lam * degree[members].max() / delta
            image_bounds.append(image_bound)
            node_steps[members] = 1. / image_bound
        image_bounds = torch.stack(image_bounds)
        current = h0
        energies = []
        for iteration in range(int(self.cfg["inner_steps"]) + 1):
            difference = current[left] - current[right]
            root = torch.sqrt(difference.square() + delta ** 2)
            energy = (0.5 * a * (current - h0).square().sum()
                      + 0.5 * eta * (c[:, None] * (current - target).square()).sum()
                      + lam * (weights[:, None] * (root - delta)).sum())
            energies.append(energy.detach())
            gradient = a * (current - h0) + eta * c[:, None] * (current - target)
            if len(edges):
                flux = lam * weights[:, None] * difference / root
                gradient = gradient.index_add(0, left, flux).index_add(0, right, -flux)
            if iteration < int(self.cfg["inner_steps"]):
                current = current - node_steps[:, None] * gradient
        diag = {"energy_history": torch.stack(energies), "edge_count": len(edges), "edges": edges,
                "step_size": node_steps.detach().min(), "lipschitz_bound": image_bounds.detach().max(),
                "image_ids": image_ids, "image_lipschitz_bounds": image_bounds.detach(),
                "node_step_sizes": node_steps.detach(),
                "gradient_residual": gradient.detach().norm(), "coefficients_fixed": True,
                "max_weighted_degree": degree.detach().max(), "strong_convexity_floor": a,
                "memory_entry": "single_soft_tether", "iterations": int(self.cfg["inner_steps"])}
        return current.to(original_dtype), diag
