# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for running an experiment."""

import functools
from typing import Any
import os

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from tqdm import tqdm


class Experiment:
    """Class to initialize and maintain experiment state."""

    def __init__(
        self,
        forward: hk.TransformedWithState,
        optimizer: optax.GradientTransformation,
        rng: jax.Array = jax.random.PRNGKey(0),
        main_process: bool = True,
    ) -> None:
        """Initializes an experiment."""
        self.forward = forward
        self.optimizer = optimizer
        self.rng = rng
        self.main_process = main_process

        self.params = None
        self.network_state = None
        self.opt_state = None

    def step(self, inputs: chex.ArrayTree) -> dict[str, jax.Array]:
        """Takes a single step of the experiment."""
        if self.params is None:
            self.init_fn(inputs)

        self.params, self.network_state, self.opt_state, self.rng, metrics = (
            self.update_fn(
                self.params,
                self.network_state,
                self.opt_state,
                self.rng,
                inputs,
            )
        )

        return metrics

    def init_fn(self, inputs: chex.ArrayTree) -> None:
        """Initializes the experiment."""
        init_fn = self.forward.init
        self.params, self.network_state = init_fn(
            self.rng, inputs["src"], is_training=True
        )
        self.opt_state = self.optimizer.init(self.params)

    def task_fn(
        self,
        outputs: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Computes the task loss, accuracy, and metrics."""
        outputs_expanded = jnp.repeat(outputs, targets.shape[1], axis=1)
        loss = jnp.mean(optax.squared_error(outputs_expanded, targets))
        preds = jnp.argmax(outputs_expanded, axis=-1)
        targets_class = jnp.argmax(targets, axis=-1)
        correct = jnp.sum((preds == targets_class).astype(jnp.float32))
        count = jnp.prod(jnp.array(targets.shape[:-2]))
        return count, loss, correct

    def loss_fn(
        self,
        params: chex.ArrayTree,
        state: chex.ArrayTree,
        rng: jax.Array,
        data: chex.ArrayTree,
        is_training: bool = True,
    ) -> tuple[jax.Array, tuple[dict[str, jax.Array], Any]]:
        """Computes the loss and metrics for a batch of data."""
        outputs, state = self.forward.apply(
            params,
            state,
            rng=rng,
            inputs=data["src"],
            is_training=is_training,
        )
        count, loss, correct = self.task_fn(outputs, data["tgt"])
        metrics = dict(count=count, loss=loss, correct=correct)
        return loss, (metrics, state)

    def update_fn(
        self,
        params: chex.ArrayTree,
        network_state: chex.ArrayTree,
        opt_state: chex.ArrayTree,
        rng: jax.Array,
        inputs: chex.ArrayTree,
    ) -> tuple[
        chex.ArrayTree,
        chex.ArrayTree,
        chex.ArrayTree,
        jax.Array,
        dict[str, jax.Array],
    ]:
        """Applies an update to parameters and returns new state."""
        rng, subrng = jax.random.split(rng)
        grad_loss_fn = jax.grad(self.loss_fn, has_aux=True)
        grads, (metrics, network_state) = grad_loss_fn(
            params, network_state, subrng, inputs
        )
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, network_state, opt_state, rng, metrics

    def eval_epoch(self, dataset: tf.data.Dataset) -> dict[str, jax.Array]:
        """Evaluates an epoch."""
        epoch_metrics = {
            "count": jnp.array(0.0),
            "loss": jnp.array(0.0),
            "correct": jnp.array(0.0),
        }
        pbar = tqdm(dataset, desc="Evaluating", disable=not self.main_process)
        for inputs in pbar:
            _, (metrics, _) = self.loss_fn(
                self.params, self.network_state, self.rng, inputs, is_training=False
            )
            for k, v in metrics.items():
                epoch_metrics[k] += v
            pbar.set_postfix({"loss": epoch_metrics["loss"] / epoch_metrics["count"]})
        return epoch_metrics

    def save_checkpoint(self, path: str) -> None:
        """Saves the model checkpoint."""
        checkpoint = {
            "params": self.params,
            "network_state": self.network_state,
            "opt_state": self.opt_state,
            "rng": self.rng,
        }
        with open(path, "wb") as f:
            jax.numpy.savez(f, **checkpoint)
        if self.main_process:
            print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Loads the model checkpoint."""
        with open(path, "rb") as f:
            checkpoint = jax.numpy.load(f)
        self.params = checkpoint["params"].item()
        self.network_state = checkpoint["network_state"].item()
        self.opt_state = checkpoint["opt_state"].item()
        self.rng = checkpoint["rng"]
        if self.main_process:
            print(f"Checkpoint loaded from {path}")
