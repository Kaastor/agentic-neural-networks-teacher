"""Backpropagation concept module with canonical closed-book content."""

from __future__ import annotations

from app.content.schema import (
    CanonicalFact,
    Concept,
    ContentSection,
    LearningObjective,
    ProblemInstance,
    ProblemTemplate,
    ProblemTemplateMetadata,
    WorkedExample,
)

CONCEPT_ID = "concept-backpropagation"

SECTIONS: list[ContentSection] = [
    ContentSection(
        id="sec-origin-story",
        title="Why Backpropagation Matters",
        body=(
            "Backpropagation is the algorithmic heartbeat of supervised deep learning. It "
            "provides the exact gradients that optimizers such as stochastic gradient descent "
            "need in order to minimize a loss function. Before backpropagation, hand-derived "
            "gradients limited the complexity of neural networks people were willing to train. "
            "This section establishes the historical motivation, highlights the breakthrough "
            "insight of reverse-mode automatic differentiation, and frames backpropagation as "
            "a dynamic programming algorithm that shares intermediate computations between "
            "the forward and backward passes."
        ),
    ),
    ContentSection(
        id="sec-math-prereqs",
        title="Mathematical Prerequisites",
        body=(
            "Backpropagation stands on multivariable calculus and linear algebra. Learners must "
            "be comfortable with gradients, Jacobians, vector-Jacobian products, and matrix "
            "multiplication. This section reviews these foundations, clarifies notation for "
            "scalars, vectors, matrices, and tensors, and revisits partial derivatives with "
            "respect to each parameter. We emphasize how broadcasting rules and tensor shapes "
            "affect the computation of gradients—an essential practical consideration when "
            "implementing the algorithm in code."
        ),
    ),
    ContentSection(
        id="sec-notation",
        title="Notation and Computational Graphs",
        body=(
            "To make backpropagation rigorous we rely on computational graphs where nodes "
            "represent operations and edges represent intermediate values. This section sets up "
            "the notation: superscripts for layer indices, subscripts for neuron indices, bold "
            "symbols for vectors, and capital letters for matrices. We walk through a toy "
            "two-layer network, drawing the forward graph and labeling the intermediate "
            "activations and pre-activations. The graph makes explicit where the chain rule will "
            "later apply."
        ),
    ),
    ContentSection(
        id="sec-forward-loss",
        title="Forward Propagation and Loss Landscapes",
        body=(
            "Before differentiating we must understand the forward computation. We formalize "
            "how inputs propagate through affine transformations and nonlinearities to produce "
            "a prediction. The section then details common loss functions such as mean squared "
            "error and cross-entropy, showing how they compose with the network output. We "
            "introduce the notion of a loss landscape over the parameter space and motivate why "
            "gradient information is the most efficient signal for optimization."
        ),
    ),
    ContentSection(
        id="sec-chain-rule",
        title="Chain Rule Refresher",
        body=(
            "Backpropagation is a systematic application of the multivariable chain rule. We "
            "derive the chain rule for composite scalar functions, extend it to vector-valued "
            "functions using Jacobians, and highlight the efficiency of reverse-mode automatic "
            "differentiation when the function has many inputs but a single output. Worked "
            "mini-examples demonstrate how gradients flow backwards by multiplying local "
            "derivatives in reverse topological order."
        ),
    ),
    ContentSection(
        id="sec-scalar-derivation",
        title="Deriving Backpropagation for a Scalar Output",
        body=(
            "We formalize the derivation for a feedforward network producing a scalar loss. The "
            "derivation begins with the final layer, computing the gradient of the loss with "
            "respect to the logits. We then push the gradient backwards through each layer by "
            "multiplying with the Jacobian of the layer's activation function and the transpose "
            "of the weight matrix. This section introduces the local error term delta and "
            "shows how it serves as a reusable summary of upstream gradients for each layer."
        ),
    ),
    ContentSection(
        id="sec-vjp",
        title="Vector-Jacobian Products and Efficiency",
        body=(
            "Reverse-mode differentiation avoids forming full Jacobian matrices by propagating "
            "vector-Jacobian products. We derive the vector-Jacobian form for linear, "
            "elementwise, and broadcasting operations. The section clarifies why reverse-mode "
            "cost scales with the number of outputs rather than inputs, making it ideal for "
            "training neural networks with millions of parameters but relatively few loss "
            "values."
        ),
    ),
    ContentSection(
        id="sec-mlp-case-study",
        title="Case Study: Two-Layer MLP",
        body=(
            "A concrete walkthrough anchors intuition. We compute the forward and backward "
            "passes for a two-layer multilayer perceptron with a ReLU hidden layer and a linear "
            "output layer trained with mean squared error. Each intermediate derivative is "
            "written explicitly, from the gradient with respect to the output weights back to "
            "the input features. The section highlights how intermediate activations are cached "
            "and reused in reverse order."
        ),
    ),
    ContentSection(
        id="sec-matrix-form",
        title="Matrix Notation and Generalization",
        body=(
            "We generalize the derivation into matrix notation, replacing index-heavy expressions "
            "with compact linear algebra. The section defines how deltas become matrices aligned "
            "with activations, demonstrates the gradient for batched inputs, and discusses "
            "broadcasting conventions used by deep learning frameworks. We also connect the "
            "matrix form to computational graphs used by automatic differentiation systems."
        ),
    ),
    ContentSection(
        id="sec-implementation",
        title="Implementation Patterns and Numerical Stability",
        body=(
            "Translating backpropagation into code requires careful handling of numerical "
            "precision. This section enumerates implementation pitfalls: activation saturation, "
            "vanishing and exploding gradients, and floating-point underflow in softmax cross "
            "entropy. Strategies such as Xavier/He initialization, gradient clipping, and log-sum-exp "
            "stabilization are discussed. We emphasize unit testing of gradient computations and "
            "the role of automatic differentiation libraries."
        ),
    ),
    ContentSection(
        id="sec-optimization",
        title="Interplay with Optimization Algorithms",
        body=(
            "Backpropagation only supplies gradients; optimization algorithms determine how to "
            "use them. We outline how optimizers such as SGD with momentum, RMSProp, and Adam "
            "consume gradient information, highlighting their assumptions about gradient noise "
            "and curvature. Learners see how poor gradient quality directly degrades convergence, "
            "motivating rigorous gradient validation."
        ),
    ),
    ContentSection(
        id="sec-diagnostics",
        title="Diagnostics and Common Pitfalls",
        body=(
            "The final section catalogs frequent backpropagation bugs: incorrect tensor shapes, "
            "forgetting to reset gradients, mixing row- and column-major conventions, and "
            "mismatched loss/activation pairs. We provide checklists for debugging, including "
            "gradient norm logging, finite-difference checks, and visualization of activation "
            "distributions layer by layer."
        ),
    ),
]

LEARNING_OBJECTIVES: list[LearningObjective] = [
    LearningObjective(
        id="lo-bp-derive-two-layer",
        concept_id=CONCEPT_ID,
        bloom="analyze",
        difficulty="intermediate",
        statement="Derive the gradient updates for weights and biases in a two-layer MLP trained with mean squared error.",
        assessment_methods=["derivation", "coding", "oral-explanation"],
        prerequisites=["calc-chain-rule", "lin-alg-matrix-mult"],
    ),
    LearningObjective(
        id="lo-bp-jacobian-intuition",
        concept_id=CONCEPT_ID,
        bloom="understand",
        difficulty="introductory",
        statement="Explain how vector-Jacobian products enable efficient reverse-mode automatic differentiation.",
        assessment_methods=["concept-check", "short-answer"],
        prerequisites=["calc-multivariable-basics"],
    ),
    LearningObjective(
        id="lo-bp-debug",
        concept_id=CONCEPT_ID,
        bloom="evaluate",
        difficulty="advanced",
        statement="Diagnose and correct implementation bugs that lead to vanishing or exploding gradients in multilayer perceptrons.",
        assessment_methods=["code-review", "case-study"],
        prerequisites=["lo-bp-derive-two-layer"],
    ),
    LearningObjective(
        id="lo-bp-softmax-ce",
        concept_id=CONCEPT_ID,
        bloom="apply",
        difficulty="intermediate",
        statement="Compute gradients of the softmax-cross-entropy loss with respect to logits in classification networks.",
        assessment_methods=["derivation", "numerical"],
        prerequisites=["calc-vector-derivatives"],
    ),
    LearningObjective(
        id="lo-bp-gradient-check",
        concept_id=CONCEPT_ID,
        bloom="create",
        difficulty="advanced",
        statement="Implement a gradient checking routine that validates analytical gradients against finite-difference estimates.",
        assessment_methods=["coding", "reflection"],
        prerequisites=["lo-bp-derive-two-layer"],
    ),
]

CANONICAL_FACTS: list[CanonicalFact] = [
    CanonicalFact(
        id="fact-chain-rule",
        concept_id=CONCEPT_ID,
        title="Multivariable Chain Rule",
        statement=(
            "For a composite function L = f(g(x)) where f maps R^m to R and g maps R^n to R^m, "
            "the gradient of L with respect to x equals the vector-Jacobian product grad_x L = J_g(x)^T grad_g f."
        ),
        derivation=(
            "Using directional derivatives we express dL = grad_g f dot dg. Because dg = J_g(x) dx, "
            "substituting yields dL = grad_g f dot J_g(x) dx. Collecting terms shows grad_x L = J_g(x)^T grad_g f."
        ),
        references=["Calc III lecture notes", "Backpropagation revisited - Rumelhart et al., 1986"],
        related_objectives=["lo-bp-derive-two-layer", "lo-bp-softmax-ce"],
        pitfalls=["Confusing Jacobian dimensions", "Multiplying in incorrect order"],
    ),
    CanonicalFact(
        id="fact-delta-recursion",
        concept_id=CONCEPT_ID,
        title="Delta Recurrence",
        statement=(
            "For layer l with activation h^l, weights W^l, and elementwise nonlinearity sigma^l, the error term "
            "delta^l = (W^{l+1})^T delta^{l+1} hadamard sigma_prime^l(z^l)."
        ),
        derivation=(
            "Applying the chain rule to h^l = sigma^l(z^l) and z^l = W^l h^{l-1} + b^l shows that the gradient with respect "
            "to h^{l-1} requires multiplying the upstream delta by W^l. Re-indexing for a general layer yields the "
            "recursive form where the derivative of the nonlinearity gates the upstream signal."
        ),
        references=["Deep Learning (Goodfellow et al.) Section 6.5"],
        related_objectives=["lo-bp-derive-two-layer"],
        pitfalls=["Dropping the elementwise product", "Transposing W incorrectly"],
    ),
    CanonicalFact(
        id="fact-softmax-gradient",
        concept_id=CONCEPT_ID,
        title="Softmax Cross-Entropy Gradient",
        statement=(
            "For logits z and one-hot targets y, the gradient of cross-entropy L = -sum_i y_i log softmax(z)_i "
            "with respect to z is softmax(z) - y."
        ),
        derivation=(
            "We first compute dL/dz_i using the quotient rule on the exponential normalization. Because the Jacobian of "
            "softmax has a structured form, contracting with the cross-entropy gradient simplifies to the difference "
            "between predicted probabilities and the target distribution."
        ),
        references=["CS231n Notes", "Pattern Recognition and Machine Learning"],
        related_objectives=["lo-bp-softmax-ce"],
        pitfalls=["Forgetting to subtract y", "Not handling batch dimension"],
    ),
    CanonicalFact(
        id="fact-gradient-check",
        concept_id=CONCEPT_ID,
        title="Central Difference Gradient Check",
        statement=(
            "Finite-difference gradient checks compare the analytical gradient g to the numerical estimate "
            "g_hat_i = (L(theta_i + epsilon) - L(theta_i - epsilon)) / (2 * epsilon)."
        ),
        derivation=(
            "Using the second-order Taylor expansion of L around theta_i shows that the central difference estimator "
            "approximates the true derivative with error on the order of epsilon squared."
        ),
        references=["Numerical Optimization - Nocedal & Wright"],
        related_objectives=["lo-bp-gradient-check"],
        pitfalls=["Choosing epsilon too large", "Failing to disable regularization during check"],
    ),
    CanonicalFact(
        id="fact-gradients-bias",
        concept_id=CONCEPT_ID,
        title="Bias Gradient",
        statement=(
            "For bias vector b^l in layer l, the gradient is the sum of the error term over the batch: grad_{b^l} L = sum_k delta^l_k."
        ),
        derivation=(
            "Because z^l = W^l h^{l-1} + b^l, differentiating with respect to b^l adds one for each example. The gradient "
            "therefore accumulates the layer's delta across examples."
        ),
        references=["Deep Learning (Goodfellow et al.) Section 6.5"],
        related_objectives=["lo-bp-derive-two-layer"],
        pitfalls=["Averaging instead of summing when optimizer expects mean"],
    ),
]

WORKED_EXAMPLES: list[WorkedExample] = [
    WorkedExample(
        id="example-two-layer-derivation",
        concept_id=CONCEPT_ID,
        title="Manual Backpropagation Through a Two-Layer Network",
        narrative=(
            "We examine a two-layer MLP with ReLU activation and mean squared error loss on a single training example."
        ),
        steps=[
            "Perform the forward pass, caching z^1, h^1, z^2, and the network output y_hat.",
            "Compute the loss derivative with respect to y_hat and obtain delta^2.",
            "Propagate the error back to the hidden layer: delta^1 = (W^2)^T delta^2 hadamard indicator[z^1 > 0].",
            "Compute gradients for W^2 and b^2 using h^1 and delta^2, and for W^1 and b^1 using x and delta^1.",
            "Verify gradient magnitudes with a finite-difference check before applying an optimizer step.",
        ],
        takeaways=[
            "Each layer reuses upstream gradients through delta terms, avoiding redundant derivatives.",
            "Caching intermediate activations during the forward pass is essential for efficiency.",
        ],
    ),
    WorkedExample(
        id="example-softmax",
        concept_id=CONCEPT_ID,
        title="Softmax Cross-Entropy Backward Step",
        narrative=(
            "We compute the backward step for a classifier with a final softmax layer trained using cross-entropy."
        ),
        steps=[
            "Run the forward pass to obtain logits z and probabilities p = softmax(z).",
            "Given one-hot targets y, compute gradient grad_z L = p - y.",
            "Multiply the gradient by the hidden activations to obtain grad_W L, respecting batch dimensions.",
            "Sum over the batch to obtain bias gradients, matching the implementation expected by optimizers.",
        ],
        takeaways=[
            "Softmax and cross-entropy form a numerically stable pair whose gradients simplify elegantly.",
            "Batch dimensions must be handled explicitly to prevent silent broadcasting errors.",
        ],
    ),
    WorkedExample(
        id="example-gradient-check",
        concept_id=CONCEPT_ID,
        title="Implementing a Gradient Checker",
        narrative=(
            "We implement a gradient checker for a two-layer network and interpret its output."),
        steps=[
            "Flatten parameters into a single vector for ease of perturbation.",
            "For each parameter index, compute the central-difference estimate with a small epsilon.",
            "Compare analytical and numerical gradients using relative error frac{|g - g_hat|}{max(1, |g|, |g_hat|)}.",
            "Flag parameters whose relative error exceeds 1e-6 and inspect their computation graphs.",
        ],
        takeaways=[
            "Gradient checkers provide a safety net before large-scale experiments.",
            "Disable dropout, data augmentation, and regularization terms during gradient checking.",
        ],
    ),
]

PROBLEM_TEMPLATES: list[ProblemTemplate] = [
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-linear-mse-gradient",
            concept_id=CONCEPT_ID,
            title="Gradient of Linear Regression with MSE",
            description="Derive gradients for weights and biases in a single-layer linear model under MSE.",
            difficulty="introductory",
            problem_type="derivation",
            canonical_fact_ids=["fact-chain-rule"],
            learning_objective_ids=["lo-bp-derive-two-layer"],
            variant_ids=["v1"],
            tags=["foundations", "matrix-calculus"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-linear-mse-gradient",
                variant_id="v1",
                prompt=(
                    "Consider y = Wx + b with loss L = 0.5 ||y - t||_2^2 for a single training example. "
                    "Derive expressions for grad_W L and grad_b L in terms of x, t, and the prediction y."
                ),
                solution=(
                    "Because L = 0.5 (y - t)^T (y - t), we have grad_y L = y - t. Using the chain rule, "
                    "grad_W L = (y - t) x^T and grad_b L = y - t."
                ),
                answer="grad_W L = (y - t) x^T, grad_b L = y - t",
                rubric="Check that the learner differentiates with respect to the pre-activation, not the loss directly.",
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-relu-derivative",
            concept_id=CONCEPT_ID,
            title="Derivative of ReLU",
            description="Explain and compute the derivative of the ReLU activation within a network context.",
            difficulty="introductory",
            problem_type="conceptual",
            canonical_fact_ids=["fact-delta-recursion"],
            learning_objective_ids=["lo-bp-derive-two-layer"],
            variant_ids=["v1"],
            tags=["activation"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-relu-derivative",
                variant_id="v1",
                prompt=(
                    "A hidden neuron applies ReLU to its pre-activation z. During backpropagation the upstream gradient "
                    "is g. Write the local gradient and final gradient contribution to the neuron's input."
                ),
                solution="The derivative is 1 when z > 0 and 0 otherwise. The contribution is g if z > 0 else 0.",
                answer="g * 1[z > 0]",
                rubric="Learner must articulate the gating behavior and mention the stored pre-activation sign.",
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-softmax-ce",
            concept_id=CONCEPT_ID,
            title="Softmax Cross-Entropy Gradient",
            description="Compute the gradient of the softmax cross-entropy loss with respect to logits.",
            difficulty="intermediate",
            problem_type="derivation",
            canonical_fact_ids=["fact-softmax-gradient"],
            learning_objective_ids=["lo-bp-softmax-ce"],
            variant_ids=["v1"],
            tags=["classification", "softmax"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-softmax-ce",
                variant_id="v1",
                prompt=(
                    "Given logits z and one-hot target vector y, derive grad_z L for cross-entropy loss "
                    "L = -sum_i y_i log softmax(z)_i."
                ),
                solution="The gradient is softmax(z) - y.",
                answer="softmax(z) - y",
                rubric="Learner should justify how the Jacobian structure leads to subtraction of y.",
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-delta-recursion",
            concept_id=CONCEPT_ID,
            title="Hidden Layer Delta",
            description="Show how to backpropagate the error through a hidden layer with sigmoid activation.",
            difficulty="intermediate",
            problem_type="derivation",
            canonical_fact_ids=["fact-delta-recursion"],
            learning_objective_ids=["lo-bp-derive-two-layer"],
            variant_ids=["v1"],
            tags=["hidden-layer", "sigmoid"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-delta-recursion",
                variant_id="v1",
                prompt=(
                    "Layer l uses sigmoid activation. Given delta^{l+1}, weights W^{l+1}, and cached activation h^l, derive delta^l."
                ),
                solution=(
                    "Compute sigma'(z^l) = h^l hadamard (1 - h^l). Then delta^{l} = (W^{l+1})^T delta^{l+1} hadamard sigma'(z^l)."
                ),
                answer="(W^{l+1})^T delta^{l+1} hadamard h^l (1 - h^l)",
                rubric="Check use of h^l instead of z^l when computing sigmoid derivative.",
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-gradient-check",
            concept_id=CONCEPT_ID,
            title="Gradient Checker Implementation",
            description="Implement a gradient checker for a network using central differences.",
            difficulty="advanced",
            problem_type="coding",
            canonical_fact_ids=["fact-gradient-check"],
            learning_objective_ids=["lo-bp-gradient-check"],
            variant_ids=["v1"],
            tags=["tooling", "diagnostics"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-gradient-check",
                variant_id="v1",
                prompt=(
                    "Write a Python function `gradient_check(model, loss_fn, params, epsilon=1e-5)` that compares analytical "
                    "gradients from `model.backward` with central-difference estimates. Return the maximum relative error."),
                solution=(
                    "Iterate over flattened parameters, perturb +epsilon and -epsilon, compute losses, and form the "
                    "central difference. Compare to the analytical gradient and track the maximum relative error."),
                rubric="Unit tests expect iteration over parameters, central difference, and reporting of max relative error.",
                unit_test_stub=(
                    "def test_gradient_check_returns_small_error():\n"
                    "    error = gradient_check(model, loss_fn, params)\n"
                    "    assert error < 1e-6\n"
                ),
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-backprop-debugging",
            concept_id=CONCEPT_ID,
            title="Diagnose Exploding Gradients",
            description="Analyze a log excerpt to diagnose exploding gradients and recommend fixes.",
            difficulty="advanced",
            problem_type="analysis",
            canonical_fact_ids=["fact-delta-recursion"],
            learning_objective_ids=["lo-bp-debug"],
            variant_ids=["v1"],
            tags=["diagnostics", "stability"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-backprop-debugging",
                variant_id="v1",
                prompt=(
                    "Training logs show gradient norm jumping from 5 to 1e6 after 300 iterations. Activations saturate and "
                    "loss becomes NaN. Identify likely causes and propose two mitigation strategies."
                ),
                solution=(
                    "Likely causes include too large a learning rate or poor initialization leading to exploding gradients. "
                    "Mitigations: apply gradient clipping, reduce learning rate, or use orthogonal initialization.") ,
                rubric="Expect identification of gradient explosion and at least two concrete mitigation tactics.",
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-batch-matrix",
            concept_id=CONCEPT_ID,
            title="Batch Gradient Expression",
            description="Write the matrix expression for batched gradients in a two-layer MLP.",
            difficulty="intermediate",
            problem_type="derivation",
            canonical_fact_ids=["fact-delta-recursion", "fact-gradients-bias"],
            learning_objective_ids=["lo-bp-derive-two-layer"],
            variant_ids=["v1"],
            tags=["matrix", "batching"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-batch-matrix",
                variant_id="v1",
                prompt=(
                    "For batch size B, derive matrix expressions for grad_W2 L and grad_W1 L in a two-layer MLP with ReLU.") ,
                solution=(
                    "With hidden activations H^1 (B x H) and deltas Delta^2 (B x O), gradients are grad_W2 L = (Delta^2)^T H^1 / B and "
                    "grad_W1 L = (Delta^1)^T X / B with Delta^1 = (Delta^2 W^2) hadamard indicator[Z^1 > 0]."
                ),
                answer="grad_W2 L = (Delta^2)^T H^1 / B, grad_W1 L = (Delta^1)^T X / B",
                rubric="Expect explicit mention of averaging over batch and the ReLU mask.",
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-code-two-layer",
            concept_id=CONCEPT_ID,
            title="Implement Two-Layer Backprop",
            description="Implement forward and backward passes for a two-layer MLP from scratch.",
            difficulty="advanced",
            problem_type="coding",
            canonical_fact_ids=["fact-delta-recursion", "fact-gradients-bias"],
            learning_objective_ids=["lo-bp-derive-two-layer"],
            variant_ids=["v1"],
            tags=["coding-lab", "mlp"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-code-two-layer",
                variant_id="v1",
                prompt=(
                    "Write a NumPy-based function `mlp_forward_backward(params, batch)` that returns predictions and gradients "
                    "for W1, b1, W2, b2 given ReLU activation and mean squared error."
                ),
                solution=(
                    "The implementation should compute z1 = X @ W1 + b1, h1 = relu(z1), z2 = h1 @ W2 + b2, predictions = z2. "
                    "Backward pass computes delta2 = (pred - y) / B, grad_W2 = h1.T @ delta2, grad_b2 = delta2.sum(axis=0), "
                    "delta1 = (delta2 @ W2.T) * (z1 > 0), grad_W1 = X.T @ delta1, grad_b1 = delta1.sum(axis=0)."
                ),
                rubric="Unit tests check tensor shapes, broadcasting, and reuse of cached activations.",
                unit_test_stub=(
                    "def test_mlp_forward_backward_shapes():\n"
                    "    preds, grads = mlp_forward_backward(params, batch)\n"
                    "    assert preds.shape == (batch['x'].shape[0], batch['y'].shape[1])\n"
                    "    for key in ('W1', 'b1', 'W2', 'b2'):\n"
                    "        assert key in grads\n"
                ),
            )
        },
    ),
    ProblemTemplate(
        metadata=ProblemTemplateMetadata(
            id="pt-loss-selection",
            concept_id=CONCEPT_ID,
            title="Match Activation and Loss",
            description="Select compatible activation and loss pairs to ensure stable gradients.",
            difficulty="introductory",
            problem_type="conceptual",
            canonical_fact_ids=["fact-softmax-gradient"],
            learning_objective_ids=["lo-bp-softmax-ce"],
            variant_ids=["v1"],
            tags=["losses", "activations"],
        ),
        variants={
            "v1": ProblemInstance(
                template_id="pt-loss-selection",
                variant_id="v1",
                prompt=(
                    "Choose appropriate loss functions for outputs using (a) linear activation, (b) sigmoid activation, (c) softmax activation. "
                    "Explain gradient stability implications."
                ),
                solution=(
                    "(a) Pair with mean squared error or L1. (b) Use binary cross-entropy ensuring gradients do not saturate. "
                    "(c) Use categorical cross-entropy; pairing maintains stable gradients via softmax - target difference."
                ),
                rubric="Expect correct pairings and explicit reference to gradient behavior.",
            )
        },
    ),
]

CONCEPT = Concept(
    id=CONCEPT_ID,
    slug="backpropagation",
    title="Backpropagation",
    summary=(
        "Comprehensive closed-book module covering the derivation, intuition, and implementation of backpropagation "
        "for multilayer perceptrons, including diagnostics, canonical facts, and coding labs."
    ),
    prerequisites=["concept-linear-algebra", "concept-gradient-descent"],
    primary_objectives=[lo.id for lo in LEARNING_OBJECTIVES],
    canonical_fact_ids=[fact.id for fact in CANONICAL_FACTS],
    worked_example_ids=[example.id for example in WORKED_EXAMPLES],
    problem_template_ids=[template.metadata.id for template in PROBLEM_TEMPLATES],
    sections=SECTIONS,
)
