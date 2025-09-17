"""Integration tests for the math verification service."""

from __future__ import annotations

from typing import cast

import numpy as np
from fastapi.testclient import TestClient

from services.math_service.main import app

client = TestClient(app)


def test_symbolic_equality_detects_equivalent_expressions() -> None:
    """Expressions that simplify to the same polynomial are accepted."""

    payload = {
        "canonical": "x**2 + 2*x + 1",
        "candidate": "(x + 1)**2",
        "symbols": ["x"],
        "random_trials": 3,
    }
    response = client.post("/check/equality", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["equivalent"] is True
    assert "identical" in body["detail"]


def test_symbolic_equality_flags_mismatched_expressions() -> None:
    """Clearly mismatched expressions are rejected."""

    payload = {
        "canonical": "sin(x)",
        "candidate": "cos(x)",
        "symbols": ["x"],
        "random_trials": 2,
    }
    response = client.post("/check/equality", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["equivalent"] is False


def test_jacobian_check_passes_for_mlp_gradient() -> None:
    """Analytic Jacobian of a two-layer tanh MLP matches the service's reference."""

    function = [
        "w2_11*tanh(w1_11*x1 + w1_12*x2 + b1_1) + w2_12*tanh(w1_21*x1 + w1_22*x2 + b1_2) + b2_1",
        "w2_21*tanh(w1_11*x1 + w1_12*x2 + b1_1) + w2_22*tanh(w1_21*x1 + w1_22*x2 + b1_2) + b2_2",
    ]
    variables = ["x1", "x2"]
    jacobian = [
        [
            "w2_11*(1 - tanh(w1_11*x1 + w1_12*x2 + b1_1)**2)*w1_11 + "
            "w2_12*(1 - tanh(w1_21*x1 + w1_22*x2 + b1_2)**2)*w1_21",
            "w2_11*(1 - tanh(w1_11*x1 + w1_12*x2 + b1_1)**2)*w1_12 + "
            "w2_12*(1 - tanh(w1_21*x1 + w1_22*x2 + b1_2)**2)*w1_22",
        ],
        [
            "w2_21*(1 - tanh(w1_11*x1 + w1_12*x2 + b1_1)**2)*w1_11 + "
            "w2_22*(1 - tanh(w1_21*x1 + w1_22*x2 + b1_2)**2)*w1_21",
            "w2_21*(1 - tanh(w1_11*x1 + w1_12*x2 + b1_1)**2)*w1_12 + "
            "w2_22*(1 - tanh(w1_21*x1 + w1_22*x2 + b1_2)**2)*w1_22",
        ],
    ]

    payload = {
        "function": function,
        "variables": variables,
        "candidate": jacobian,
        "evaluation_point": {
            "x1": 0.25,
            "x2": -0.5,
            "w1_11": 0.8,
            "w1_12": -0.3,
            "w1_21": 0.1,
            "w1_22": 0.7,
            "w2_11": 1.1,
            "w2_12": -1.3,
            "w2_21": 0.55,
            "w2_22": -0.45,
            "b1_1": 0.2,
            "b1_2": -0.4,
            "b2_1": 0.05,
            "b2_2": -0.12,
        },
        "random_trials": 2,
    }
    response = client.post("/check/derivative", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["equivalent"] is True


def test_mlp_gradient_matches_numerical_check() -> None:
    """Analytic MLP Jacobian aligns with a finite-difference gradient harness."""

    rng = np.random.default_rng(seed=42)
    x = rng.normal(size=(2,))
    w1 = rng.normal(size=(2, 2))
    b1 = rng.normal(size=(2,))
    w2 = rng.normal(size=(2, 2))
    b2 = rng.normal(size=(2,))

    def forward(vec: np.ndarray) -> np.ndarray:
        hidden = np.tanh(w1 @ vec + b1)
        return cast(np.ndarray, w2 @ hidden + b2)

    def analytic_jacobian(vec: np.ndarray) -> np.ndarray:
        pre_activation = w1 @ vec + b1
        diag = np.diag(1.0 - np.tanh(pre_activation) ** 2)
        return cast(np.ndarray, w2 @ diag @ w1)

    def numerical_jacobian(vec: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        jac = np.zeros((2, 2))
        for i in range(vec.size):
            perturb = np.zeros_like(vec)
            perturb[i] = epsilon
            upper = forward(vec + perturb)
            lower = forward(vec - perturb)
            jac[:, i] = (upper - lower) / (2 * epsilon)
        return jac

    analytic = analytic_jacobian(x)
    numeric = numerical_jacobian(x)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_jacobian_check_rejects_incorrect_gradient() -> None:
    """Incorrect Jacobian entries trigger a rejection."""

    payload = {
        "function": ["x1**2 + x2", "x1 + x2**2"],
        "variables": ["x1", "x2"],
        "candidate": [["2*x1", "1"], ["1", "2*x2 + 1"]],
        "evaluation_point": {"x1": 1.0, "x2": 2.0},
        "random_trials": 1,
    }
    response = client.post("/check/derivative", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["equivalent"] is False
    assert "mismatch" in body["detail"]
