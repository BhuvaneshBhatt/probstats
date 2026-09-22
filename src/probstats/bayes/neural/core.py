"""JAX neural-network utilities for approximate Bayesian regression.

The layer combines Monte Carlo dropout, Gaussian regression heads that predict
location and precision, alpha-divergence objectives, and Lp parameter penalties in
a small JAX implementation that does not require Flax or Optax.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import InferenceKind, InferenceResult, InferenceStep


class JAXUnavailableError(ImportError):
    """Raised when a JAX-only operation is requested without JAX installed."""


def _jax():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
        raise JAXUnavailableError(
            "The neural-network layer requires JAX. Install probstats[jax]."
        ) from exc
    return jax, jnp, jsp


def _activation(name_or_callable: str | Callable[[Any], Any]) -> Callable[[Any], Any]:
    if callable(name_or_callable):
        return name_or_callable
    jax, jnp, _ = _jax()
    name = str(name_or_callable).lower().replace("_", "").replace("-", "")
    table = {
        "selu": jax.nn.selu,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "identity": lambda x: x,
        "linear": lambda x: x,
    }
    if name not in table:
        raise ValueError(f"Unknown activation {name_or_callable!r}")
    return table[name]


def _resolve_per_layer(value: Any, depth: int, *, name: str) -> tuple[Any, ...]:
    if callable(value):
        return tuple(value(i + 1) for i in range(depth))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != depth:
            raise ValueError(f"{name} sequence must have length {depth}")
        return tuple(value)
    return tuple(value for _ in range(depth))


@dataclass(frozen=True, slots=True)
class RegressionNetworkConfig:
    """Configuration for a dense Bayesian regression network."""

    input_dim: int
    depth: int = 4
    layer_size: int | Sequence[int] | Callable[[int], int] = 100
    activation: str | Callable[[Any], Any] | Sequence[Any] = "selu"
    dropout_probability: float | Sequence[float] | Callable[[int], float] | None = 0.25
    batch_normalization: bool = False
    error_model: str = "heteroscedastic"
    batchnorm_momentum: float = 0.95
    batchnorm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.depth < 0:
            raise ValueError("depth cannot be negative")
        if self.error_model.lower() not in {"heteroscedastic", "homoscedastic"}:
            raise ValueError("error_model must be 'heteroscedastic' or 'homoscedastic'")
        if not 0 <= self.batchnorm_momentum < 1:
            raise ValueError("batchnorm_momentum must lie in [0, 1)")

    @property
    def widths(self) -> tuple[int, ...]:
        values = _resolve_per_layer(self.layer_size, self.depth, name="layer_size")
        widths = tuple(int(v) for v in values)
        if any(v < 1 for v in widths):
            raise ValueError("layer sizes must be positive")
        return widths

    @property
    def dropouts(self) -> tuple[float, ...]:
        value = 0.0 if self.dropout_probability is None else self.dropout_probability
        values = _resolve_per_layer(value, self.depth, name="dropout_probability")
        result = tuple(float(v) for v in values)
        if any(not 0 <= v < 1 for v in result):
            raise ValueError("dropout probabilities must lie in [0, 1)")
        return result

    @property
    def activations(self) -> tuple[Any, ...]:
        if isinstance(self.activation, Sequence) and not isinstance(
            self.activation, (str, bytes)
        ):
            if len(self.activation) != self.depth:
                raise ValueError(f"activation sequence must have length {self.depth}")
            return tuple(self.activation)
        return tuple(self.activation for _ in range(self.depth))


@dataclass(frozen=True, slots=True)
class NeuralNetworkState:
    params: Any
    batch_stats: Any = ()


@dataclass(frozen=True, slots=True)
class NeuralPredictive:
    """Moment-matched Gaussian predictive distribution from MC dropout."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    epistemic_variance: np.ndarray
    aleatoric_variance: np.ndarray
    samples: np.ndarray

    @property
    def variance(self) -> np.ndarray:
        return self.standard_deviation**2


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    losses: tuple[float, ...]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


@dataclass(frozen=True, slots=True)
class NeuralNetworkFit:
    network: JAXRegressionNetwork
    state: NeuralNetworkState
    history: TrainingHistory
    x: np.ndarray
    y: np.ndarray
    regularization: float
    alpha: float

    def predict(
        self,
        x: Any,
        *,
        samples: int = 100,
        rng: int | None = 0,
    ) -> NeuralPredictive:
        return sample_trained_network(
            self.network, self.state, x, samples=samples, rng=rng
        )

    @property
    def training_objective(self) -> float:
        return -self.history.final_loss

    def to_inference_result(self) -> InferenceResult:
        return InferenceResult(
            posterior=self,
            kind=InferenceKind.APPROXIMATE,
            log_evidence=None,
            steps=(
                InferenceStep(
                    method="jax-mc-dropout-regression",
                    description=(
                        "Optimized a Gaussian regression network with dropout and Lp regularization; "
                        "posterior predictive uncertainty is approximated by Monte Carlo dropout."
                    ),
                    exact=False,
                    metadata={
                        "alpha": self.alpha,
                        "regularization": self.regularization,
                    },
                ),
            ),
            diagnostics={
                "training_loss": self.history.final_loss,
                "epochs": len(self.history.losses),
                "approximation": "mc-dropout",
            },
            metadata={"network": self.network, "state": self.state},
        )


class JAXRegressionNetwork:
    """Small fully connected Gaussian regression network implemented directly in JAX."""

    def __init__(self, config: RegressionNetworkConfig):
        self.config = config

    def initialize(self, rng=0) -> NeuralNetworkState:
        jax, jnp, _ = _jax()
        key = jax.random.PRNGKey(rng) if isinstance(rng, (int, np.integer)) else rng
        widths = self.config.widths
        keys = jax.random.split(key, self.config.depth + 2)
        hidden = []
        stats = []
        previous = self.config.input_dim
        for i, width in enumerate(widths):
            # LeCun normal is the natural initialization for SELU and remains sensible
            # for the other supported dense activations.
            scale = 1.0 / math.sqrt(previous)
            w = jax.random.normal(keys[i], (previous, width)) * scale
            b = jnp.zeros((width,))
            layer = {"w": w, "b": b}
            if self.config.batch_normalization:
                layer.update({"gamma": jnp.ones((width,)), "beta": jnp.zeros((width,))})
                stats.append({"mean": jnp.zeros((width,)), "var": jnp.ones((width,))})
            hidden.append(layer)
            previous = width
        output_width = 2 if self.config.error_model.lower() == "heteroscedastic" else 1
        output = {
            "w": jax.random.normal(keys[-2], (previous, output_width))
            / math.sqrt(previous),
            "b": jnp.zeros((output_width,)),
        }
        params: dict[str, Any] = {"hidden": tuple(hidden), "output": output}
        if self.config.error_model.lower() == "homoscedastic":
            params["log_precision"] = jnp.asarray(0.0)
        return NeuralNetworkState(params=params, batch_stats=tuple(stats))

    def apply(
        self,
        state: NeuralNetworkState,
        x: Any,
        *,
        training: bool = False,
        rng=None,
        update_batch_stats: bool = False,
    ) -> tuple[Any, NeuralNetworkState]:
        jax, jnp, _ = _jax()
        values = jnp.asarray(x, dtype=float)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected input shape (..., {self.config.input_dim}); got {tuple(values.shape)}"
            )
        if training and any(p > 0 for p in self.config.dropouts) and rng is None:
            raise ValueError("training/dropout evaluation requires a JAX PRNG key")
        dropout_keys = (
            jax.random.split(rng, max(self.config.depth, 1))
            if rng is not None
            else (None,) * max(self.config.depth, 1)
        )
        new_stats = []
        h = values
        for index, (layer, activation_name, pdrop) in enumerate(
            zip(state.params["hidden"], self.config.activations, self.config.dropouts)
        ):
            h = h @ layer["w"] + layer["b"]
            if self.config.batch_normalization:
                old = state.batch_stats[index]
                if training and update_batch_stats:
                    mean = jnp.mean(h, axis=0)
                    var = jnp.var(h, axis=0)
                    momentum = self.config.batchnorm_momentum
                    new_stats.append(
                        {
                            "mean": momentum * old["mean"] + (1 - momentum) * mean,
                            "var": momentum * old["var"] + (1 - momentum) * var,
                        }
                    )
                    use_mean, use_var = mean, var
                else:
                    new_stats.append(old)
                    use_mean, use_var = old["mean"], old["var"]
                h = (h - use_mean) / jnp.sqrt(use_var + self.config.batchnorm_epsilon)
                h = h * layer["gamma"] + layer["beta"]
            h = _activation(activation_name)(h)
            if training and pdrop > 0:
                keep = 1.0 - pdrop
                mask = jax.random.bernoulli(dropout_keys[index], keep, h.shape)
                h = jnp.where(mask, h / keep, 0.0)
        output_layer = state.params["output"]
        prediction = h @ output_layer["w"] + output_layer["b"]
        if self.config.error_model.lower() == "homoscedastic":
            lp = jnp.broadcast_to(state.params["log_precision"], prediction.shape)
            prediction = jnp.concatenate([prediction, lp], axis=-1)
        new_state = NeuralNetworkState(
            params=state.params,
            batch_stats=tuple(new_stats) if self.config.batch_normalization else (),
        )
        return (prediction[0] if squeeze else prediction), new_state

    def infer(
        self,
        x: Any,
        y: Any,
        **kwargs: Any,
    ) -> InferenceResult:
        return fit_regression_network(self, x, y, **kwargs).to_inference_result()


def regression_network(
    input_dim: int,
    *,
    network_depth: int = 4,
    layer_size: Any = 100,
    activation_function: Any = "selu",
    dropout_probability: Any = 0.25,
    batch_normalization: bool = False,
    error_model: str = "heteroscedastic",
) -> JAXRegressionNetwork:
    """Construct a dense Bayesian regression network."""
    return JAXRegressionNetwork(
        RegressionNetworkConfig(
            input_dim=input_dim,
            depth=network_depth,
            layer_size=layer_size,
            activation=activation_function,
            dropout_probability=dropout_probability,
            batch_normalization=batch_normalization,
            error_model=error_model,
        )
    )


def gaussian_loss(
    observed: Any,
    predicted: Any,
    *,
    parameterization: str = "log_precision",
) -> Any:
    """Gaussian deviance, equal to twice NLL up to an additive constant.

    ``predicted`` must contain the location and requested scale parameter in its
    final dimension.
    """
    _, jnp, _ = _jax()
    y = jnp.asarray(observed)
    pred = jnp.asarray(predicted)
    if pred.shape[-1] != 2:
        raise ValueError(
            "predicted must have final dimension 2: [mean, scale_parameter]"
        )
    mean, scale = pred[..., 0], pred[..., 1]
    residual2 = (mean - y) ** 2
    key = parameterization.lower().replace("_", "")
    if key in {"logprecision", "precisionlog"}:
        return residual2 * jnp.exp(scale) - scale
    if key == "variance":
        return residual2 / scale + jnp.log(scale)
    if key in {"standarddeviation", "stdev", "std"}:
        return residual2 / (scale**2) + 2 * jnp.log(scale)
    raise ValueError(f"Unknown Gaussian scale parameterization {parameterization!r}")


def gaussian_negative_log_likelihood(observed: Any, predicted: Any) -> Any:
    """external reference implementationlly normalized Gaussian negative log-likelihood per observation."""
    _jax()
    return 0.5 * (gaussian_loss(observed, predicted) + math.log(2 * math.pi))


def log_mean_exp(values: Any, axis: int | tuple[int, ...] | None = None) -> Any:
    _, jnp, jsp = _jax()
    x = jnp.asarray(values)
    if axis is None:
        count = x.size
    elif isinstance(axis, tuple):
        count = math.prod(x.shape[a] for a in axis)
    else:
        count = x.shape[axis]
    return jsp.special.logsumexp(x, axis=axis) - jnp.log(count)


def alpha_divergence_loss(
    losses: Any, alpha: float = 0.5, *, axis: int | None = None
) -> Any:
    """Evaluate the alpha-divergence loss with stable log-mean-exp arithmetic."""
    _, jnp, _ = _jax()
    x = jnp.asarray(losses)
    if alpha == 0:
        return jnp.mean(x, axis=axis)
    if alpha == math.inf:
        return jnp.min(x, axis=axis)
    if alpha == -math.inf:
        return jnp.max(x, axis=axis)
    return -log_mean_exp(-float(alpha) * x, axis=axis) / float(alpha)


def network_regularization_loss(
    params: Any,
    lambda_: float | Sequence[float] = 1.0,
    p: float | Sequence[float] = 2.0,
) -> Any:
    """Lp penalty over learned network arrays."""
    jax, jnp, _ = _jax()
    leaves = jax.tree_util.tree_leaves(params)
    if isinstance(lambda_, Sequence) or isinstance(p, Sequence):
        lambdas = tuple(lambda_) if isinstance(lambda_, Sequence) else (float(lambda_),)
        powers = tuple(p) if isinstance(p, Sequence) else (float(p),)
        if len(lambdas) != len(powers):
            raise ValueError("lambda_ and p sequences must have the same length")
        return sum(
            network_regularization_loss(params, l, power)
            for l, power in zip(lambdas, powers)
        )
    power = float(p)
    coefficient = float(lambda_)
    if power == 0:
        return jnp.asarray(
            coefficient * sum(np.prod(np.shape(leaf)) for leaf in leaves)
        )
    return coefficient * sum(jnp.sum(jnp.abs(leaf) ** power) for leaf in leaves)


def _mc_predictions(
    network: JAXRegressionNetwork,
    state: NeuralNetworkState,
    x: Any,
    *,
    samples: int,
    key: Any,
) -> Any:
    jax, jnp, _ = _jax()
    keys = jax.random.split(key, samples)
    outputs = []
    for draw_key in keys:
        pred, _ = network.apply(
            state,
            x,
            training=True,
            rng=draw_key,
            update_batch_stats=False,
        )
        outputs.append(pred)
    return jnp.stack(outputs, axis=0)


def sample_trained_network(
    network: JAXRegressionNetwork,
    state: NeuralNetworkState,
    x: Any,
    *,
    samples: int = 100,
    rng=0,
) -> NeuralPredictive:
    """Estimate predictive moments by Monte Carlo dropout."""
    if samples < 1:
        raise ValueError("samples must be positive")
    jax, jnp, _ = _jax()
    key = jax.random.PRNGKey(rng) if isinstance(rng, (int, np.integer)) else rng
    predictions = _mc_predictions(network, state, x, samples=samples, key=key)
    means = predictions[..., 0]
    log_precision = predictions[..., 1]
    mean = jnp.mean(means, axis=0)
    epistemic = jnp.var(means, axis=0)
    aleatoric = jnp.mean(jnp.exp(-log_precision), axis=0)
    std = jnp.sqrt(epistemic + aleatoric)
    return NeuralPredictive(
        mean=np.asarray(mean),
        standard_deviation=np.asarray(std),
        epistemic_variance=np.asarray(epistemic),
        aleatoric_variance=np.asarray(aleatoric),
        samples=np.asarray(predictions),
    )


def extract_regression_network(
    obj: Any,
) -> tuple[JAXRegressionNetwork, NeuralNetworkState]:
    """Return the regression network/state from a fit or frozen wrapper."""
    if isinstance(obj, NeuralNetworkFit):
        return obj.network, obj.state
    if isinstance(obj, FrozenRegressionNetwork):
        return obj.network, obj.state
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], JAXRegressionNetwork)
    ):
        return obj[0], obj[1]
    raise TypeError(
        "Expected NeuralNetworkFit, FrozenRegressionNetwork, or (network, state)"
    )


def regression_loss(
    network: JAXRegressionNetwork,
    state: NeuralNetworkState,
    x: Any,
    y: Any,
    *,
    alpha: float = 0.0,
    samples: int = 1,
    rng=0,
) -> Any:
    """Evaluate the Gaussian/alpha-divergence regression objective."""
    jax, jnp, _ = _jax()
    key = jax.random.PRNGKey(rng) if isinstance(rng, (int, np.integer)) else rng
    predictions = _mc_predictions(network, state, x, samples=samples, key=key)
    losses = gaussian_loss(jnp.asarray(y), predictions)
    return jnp.mean(alpha_divergence_loss(losses, alpha, axis=0))


def network_objective(
    network: JAXRegressionNetwork,
    state: NeuralNetworkState,
    x: Any,
    y: Any,
    lambda_: float = 1.0,
    *,
    alpha: float = 0.5,
    samples: int = 100,
    rng=0,
) -> Any:
    """Evaluate the regularized Monte Carlo dropout training objective."""
    jax, jnp, _ = _jax()
    key = jax.random.PRNGKey(rng) if isinstance(rng, (int, np.integer)) else rng
    predictions = _mc_predictions(network, state, x, samples=samples, key=key)
    losses = gaussian_loss(jnp.asarray(y), predictions)
    data_loss = jnp.mean(alpha_divergence_loss(losses, alpha, axis=0))
    regularization = network_regularization_loss(state.params, lambda_, 2)
    return -(data_loss + regularization)


def flatten_parameters(
    state: NeuralNetworkState,
) -> tuple[np.ndarray, Callable[[Any], NeuralNetworkState]]:
    """Flatten JAX parameters for use by generic Laplace/numerical backends."""
    _, _, _ = _jax()
    from jax.flatten_util import ravel_pytree

    flat, unravel = ravel_pytree(state.params)

    def restore(values: Any) -> NeuralNetworkState:
        return NeuralNetworkState(unravel(values), state.batch_stats)

    return np.asarray(flat), restore


def _adam_init(params: Any) -> tuple[Any, Any, int]:
    jax, jnp, _ = _jax()
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros, 0


def _adam_update(
    params: Any, grads: Any, state: tuple[Any, Any, int], learning_rate: float
):
    jax, jnp, _ = _jax()
    m, v, step = state
    step += 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(lambda a, g: beta1 * a + (1 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda a, g: beta2 * a + (1 - beta2) * g * g, v, grads)
    mhat = jax.tree_util.tree_map(lambda a: a / (1 - beta1**step), m)
    vhat = jax.tree_util.tree_map(lambda a: a / (1 - beta2**step), v)
    params = jax.tree_util.tree_map(
        lambda p0, a, b: p0 - learning_rate * a / (jnp.sqrt(b) + eps),
        params,
        mhat,
        vhat,
    )
    return params, (m, v, step)


def fit_regression_network(
    network: JAXRegressionNetwork,
    x: Any,
    y: Any,
    *,
    state: NeuralNetworkState | None = None,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int | None = None,
    lambda_: float = 1e-4,
    p: float = 2.0,
    alpha: float = 0.0,
    sample_number: int = 1,
    rng: int = 0,
) -> NeuralNetworkFit:
    """Train a regression network with Adam, dropout and optional alpha divergence."""
    if epochs < 1 or sample_number < 1:
        raise ValueError("epochs and sample_number must be positive")
    jax, jnp, _ = _jax()
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float).reshape(-1)
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if x_np.shape[0] != y_np.shape[0] or x_np.shape[1] != network.config.input_dim:
        raise ValueError("x/y shapes are incompatible with the network")
    current = state or network.initialize(rng)
    opt_state = _adam_init(current.params)
    generator = np.random.default_rng(rng)
    n = len(y_np)
    size = n if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be positive")
    key = jax.random.PRNGKey(rng)
    losses_history: list[float] = []

    def objective(params, batch_stats, xb, yb, step_key):
        local_state = NeuralNetworkState(params, batch_stats)
        # Update moving batch-normalization statistics once from the current batch.
        first_key, *draw_keys = jax.random.split(step_key, sample_number + 1)
        _, updated = network.apply(
            local_state,
            xb,
            training=True,
            rng=first_key,
            update_batch_stats=network.config.batch_normalization,
        )
        outputs = []
        for draw_key in draw_keys:
            pred, _ = network.apply(
                NeuralNetworkState(params, updated.batch_stats),
                xb,
                training=True,
                rng=draw_key,
                update_batch_stats=False,
            )
            outputs.append(pred)
        predictions = jnp.stack(outputs, axis=0)
        per_draw = gaussian_loss(yb, predictions)
        data_loss = jnp.mean(alpha_divergence_loss(per_draw, alpha, axis=0))
        penalty = network_regularization_loss(params, lambda_, p)
        return data_loss + penalty, updated.batch_stats

    value_and_grad = jax.value_and_grad(objective, argnums=0, has_aux=True)
    for _epoch in range(epochs):
        order = generator.permutation(n)
        epoch_values = []
        for start in range(0, n, size):
            indices = order[start : start + size]
            xb = jnp.asarray(x_np[indices])
            yb = jnp.asarray(y_np[indices])
            key, step_key = jax.random.split(key)
            (value, new_stats), grads = value_and_grad(
                current.params, current.batch_stats, xb, yb, step_key
            )
            params, opt_state = _adam_update(
                current.params, grads, opt_state, learning_rate
            )
            current = NeuralNetworkState(params, new_stats)
            epoch_values.append(float(value))
        losses_history.append(float(np.mean(epoch_values)))

    return NeuralNetworkFit(
        network=network,
        state=current,
        history=TrainingHistory(tuple(losses_history)),
        x=x_np,
        y=y_np,
        regularization=float(lambda_),
        alpha=float(alpha),
    )


def batchnorm_to_chain(
    network: JAXRegressionNetwork, state: NeuralNetworkState
) -> FrozenRegressionNetwork:
    """Freeze batch-normalization statistics while retaining stochastic dropout."""
    return FrozenRegressionNetwork(network, state)


@dataclass(frozen=True, slots=True)
class FrozenRegressionNetwork:
    network: JAXRegressionNetwork
    state: NeuralNetworkState

    def __call__(self, x: Any, *, training: bool = False, rng=None) -> Any:
        prediction, _ = self.network.apply(
            self.state,
            x,
            training=training,
            rng=rng,
            update_batch_stats=False,
        )
        return prediction
