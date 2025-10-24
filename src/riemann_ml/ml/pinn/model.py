"""Physics-informed neural network for the 1-D Euler equations."""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf

TF_EPS = tf.constant(1e-7, dtype=tf.float32)


def _ensure_float32(tensor: tf.Tensor) -> tf.Tensor:
    return tf.cast(tensor, tf.float32)


def conservative_to_primitive(
    rho: tf.Tensor, momentum: tf.Tensor, energy: tf.Tensor, gamma: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert conservative variables to primitive ones in TensorFlow."""
    rho = tf.maximum(rho, TF_EPS)
    velocity = momentum / tf.maximum(rho, TF_EPS)
    pressure = (gamma - 1.0) * (energy - 0.5 * rho * tf.square(velocity))
    pressure = tf.maximum(pressure, TF_EPS)
    return rho, velocity, pressure


def compute_flux(
    rho: tf.Tensor, momentum: tf.Tensor, energy: tf.Tensor, gamma: float
) -> tf.Tensor:
    """Compute Euler flux vector F(U) in TensorFlow."""
    rho = tf.maximum(rho, TF_EPS)
    velocity = momentum / rho
    pressure = (gamma - 1.0) * (energy - 0.5 * rho * tf.square(velocity))
    pressure = tf.maximum(pressure, TF_EPS)

    flux_mass = momentum
    flux_momentum = momentum * velocity + pressure
    flux_energy = velocity * (energy + pressure)
    return tf.concat([flux_mass, flux_momentum, flux_energy], axis=1)


def _grad_components(
    tape: tf.GradientTape, tensor: tf.Tensor, var: tf.Tensor
) -> tf.Tensor:
    """Compute gradients of each component of `tensor` with respect to `var`."""
    grads = []
    for i in range(tensor.shape[1]):
        grad = tape.gradient(tensor[:, i : i + 1], var)
        if grad is None:
            grad = tf.zeros_like(var)
        grads.append(grad)
    return tf.concat(grads, axis=1)


class PINN(tf.keras.Model):
    """Vanilla fully-connected PINN with tanh activations."""

    def __init__(self, config: Dict):
        super().__init__()
        hidden_layers = config.get("hidden_layers", 8)
        hidden_size = config.get("hidden_size", 32)
        initializer = tf.keras.initializers.GlorotUniform()

        self._layers = []
        for _ in range(hidden_layers):
            self._layers.append(
                tf.keras.layers.Dense(
                    hidden_size,
                    activation=tf.nn.tanh,
                    kernel_initializer=initializer,
                )
            )
        self._output_layer = tf.keras.layers.Dense(
            3,
            activation=None,
            kernel_initializer=initializer,
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = _ensure_float32(inputs)
        for layer in self._layers:
            x = layer(x, training=training)
        return self._output_layer(x, training=training)

    def predict_conservative(
        self, x: tf.Tensor, t: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        inputs = tf.concat([_ensure_float32(x), _ensure_float32(t)], axis=1)
        return self(inputs, training=training)

    def compute_residual(self, x: tf.Tensor, t: tf.Tensor, gamma: float) -> tf.Tensor:
        """Compute residual ∂q/∂t + ∂f(q)/∂x."""
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        x = _ensure_float32(x)
        t = _ensure_float32(t)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            inputs = tf.concat([x, t], axis=1)
            q = self(inputs, training=True)
            rho, momentum, energy = tf.split(q, 3, axis=1)
            flux = compute_flux(rho, momentum, energy, gamma)

        dq_dt = _grad_components(tape, q, t)
        df_dx = _grad_components(tape, flux, x)
        del tape
        return dq_dt + df_dx


__all__ = ["PINN", "compute_flux", "conservative_to_primitive"]
