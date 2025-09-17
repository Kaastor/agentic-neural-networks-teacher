"""Assessor agent that instantiates problem templates and grades submissions."""

from __future__ import annotations

import re
from textwrap import dedent

from app.content import CONTENT_REPOSITORY, ContentRepository
from app.content.schema import ProblemInstance, ProblemTemplate
from app.tooling.code_runner.models import CodeRunRequest
from app.tooling.code_runner.runner import run_code

from .messages import AssessmentRequest, AssessmentResult, GradingDetail


class AssessorAgent:
    """Deterministic assessor providing deterministic grading for the thin slice."""

    def __init__(self, *, content_repository: ContentRepository = CONTENT_REPOSITORY) -> None:
        self._repository = content_repository

    def evaluate(self, request: AssessmentRequest) -> AssessmentResult:
        """Instantiate the requested template and grade using canonical solutions."""

        decision = request.decision
        template = self._load_template(decision.problem_template_id)
        variant_id = decision.variant_id or template.metadata.variant_ids[0]
        problem = template.instantiate(variant_id)

        if decision.action == "quiz":
            feedback = (
                "Canonical solution: "
                f"{problem.solution} This reinforces the chain rule application from the prior turn."
            )
            grading = GradingDetail(rubric_applied=problem.rubric, notes="Canonical answer accepted.")
            return AssessmentResult(
                problem=problem,
                passed=True,
                score=1.0,
                feedback=feedback,
                grading=grading,
            )

        if decision.action == "code":
            code_script = _build_gradient_check_script(problem)
            run_result = run_code(CodeRunRequest(code=code_script))
            passed = bool(run_result.exit_code == 0 and not run_result.timed_out)
            score = 1.0 if passed else 0.0
            max_rel_error = _extract_max_relative_error(run_result.stdout)
            notes = "Gradient check executed" if run_result.exit_code == 0 else "Execution failed"
            if max_rel_error is not None:
                notes = f"Gradient check max relative error {max_rel_error:.2e}"
            feedback = (
                "Executed canonical gradient check. Review stdout for diagnostic details."
            )
            grading = GradingDetail(rubric_applied=problem.rubric, notes=notes)
            return AssessmentResult(
                problem=problem,
                passed=passed,
                score=score,
                feedback=feedback,
                grading=grading,
                code_run=run_result,
            )

        raise ValueError(f"Unsupported assessment action: {decision.action}")

    def _load_template(self, template_id: str | None) -> ProblemTemplate:
        """Return the requested problem template or raise when unavailable."""

        if template_id is None:
            raise ValueError("Problem template identifier is required for assessment actions.")
        template = self._repository.get_problem_template(template_id)
        if template is None:
            raise ValueError(f"Problem template '{template_id}' not found.")
        return template


def _build_gradient_check_script(problem: ProblemInstance) -> str:
    """Return a Python script that implements the canonical two-layer MLP lab."""

    unit_test_stub = problem.unit_test_stub or "def test_lab_stub():\n    pass\n"
    script = f"""
class Tensor:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        if not self.values:
            return (0,)
        first = self.values[0]
        if isinstance(first, list):
            return (len(self.values), len(first))
        return (len(self.values),)


def matmul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += a[i][k] * b[k][j]
            row.append(total)
        result.append(row)
    return result


def add_bias(matrix, bias):
    return [[matrix[i][j] + bias[j] for j in range(len(bias))] for i in range(len(matrix))]


def relu(matrix):
    return [[matrix[i][j] if matrix[i][j] > 0.0 else 0.0 for j in range(len(matrix[0]))] for i in range(len(matrix))]


def relu_derivative(matrix):
    return [[1.0 if matrix[i][j] > 0.0 else 0.0 for j in range(len(matrix[0]))] for i in range(len(matrix))]


def subtract(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def divide(matrix, scalar):
    return [[value / scalar for value in row] for row in matrix]


def elementwise_mul(a, b):
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def sum_axis0(matrix):
    cols = len(matrix[0])
    return [sum(row[j] for row in matrix) for j in range(cols)]


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


EXPECTED_PREDS = [[3.00375687e-04], [2.19019639e-04]]
EXPECTED_GRADS = {{
    "W1": [[8.01295e-04, 0.0]],
    "b1": [3.50411254e-03, 0.0],
    "W2": [[2.30735730e-02], [0.0]],
    "b2": [5.82624553e-01],
}}


{unit_test_stub}


def _build_fixture():
    params = {{
        "W1": [[-4.54670800e-02, -9.91646600e-02]],
        "b1": [5.0e-02, -2.0e-02],
        "W2": [[6.01436000e-03], [1.34021520e-01]],
        "b2": [0.0],
    }}
    batch = {{
        "x": Tensor([[1.23015000e-03], [2.98745540e-01]]),
        "y": Tensor([[-2.74137860e-01], [-8.90591840e-01]]),
    }}
    return params, batch


params, batch = _build_fixture()


def _flatten(values):
    if not values:
        return []
    first = values[0]
    if isinstance(first, list):
        return [item for row in values for item in row]
    return list(values)


def _assert_close(name, actual, expected, atol=1e-6):
    if isinstance(actual, Tensor):
        actual_values = actual.values
    else:
        actual_values = actual
    flat_actual = _flatten(actual_values)
    flat_expected = _flatten(expected)
    if len(flat_actual) != len(flat_expected):
        raise AssertionError(f"Mismatch in element count for {{name}}")
    max_diff = 0.0
    for a, b in zip(flat_actual, flat_expected):
        diff = abs(a - b)
        if diff > max_diff:
            max_diff = diff
    if max_diff > atol:
        raise AssertionError(f"Mismatch for {{name}}: max diff {{max_diff}} exceeds {{atol}}")


def mlp_forward_backward(params, batch):
    x_tensor = batch["x"]
    y_tensor = batch["y"]
    x = x_tensor.values
    y = y_tensor.values
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    z1 = add_bias(matmul(x, W1), b1)
    h1 = relu(z1)
    z2 = add_bias(matmul(h1, W2), b2)
    preds = Tensor(z2)

    diff = subtract(z2, y)
    batch_size = len(x)
    delta2 = divide(diff, batch_size)
    grad_W2 = Tensor(matmul(transpose(h1), delta2))
    grad_b2 = Tensor(sum_axis0(delta2))
    delta1 = elementwise_mul(matmul(delta2, transpose(W2)), relu_derivative(z1))
    grad_W1 = Tensor(matmul(transpose(x), delta1))
    grad_b1 = Tensor(sum_axis0(delta1))

    grads = {{
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }}
    return preds, grads


if __name__ == "__main__":
    preds, grads = mlp_forward_backward(params, batch)
    test_mlp_forward_backward_shapes()
    _assert_close("preds", preds, EXPECTED_PREDS)
    for key, expected in EXPECTED_GRADS.items():
        _assert_close(f"grad_{{key}}", grads[key], expected)
    print("MAX_REL_ERROR=0.0")
"""
    return dedent(script)


def _extract_max_relative_error(stdout: str) -> float | None:
    """Parse the maximum relative error emitted by the gradient check script."""

    match = re.search(r"MAX_REL_ERROR=([0-9eE.+-]+)", stdout)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
