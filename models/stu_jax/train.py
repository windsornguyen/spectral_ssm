import os
from datetime import datetime
from collections.abc import Sequence

from absl import app, flags
import jax
import haiku as hk
import jax.numpy as jnp
import tqdm

from utils.dataloader import get_dataloader, split_data
from models.stu_jax import experiment, model, optimizer
from utils.dist import setup, cleanup
from utils.colors import colored_print, Colors

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "controller", "Ant-v1", "Controller to use for the MuJoCo environment."
)
flags.DEFINE_string("task", "mujoco-v3", "Task to train on.")
flags.DEFINE_boolean("della", True, "Training on the Princeton Della cluster.")


def torch_to_jax(torch_tensor):
    """Convert a PyTorch tensor to a JAX array."""
    return jnp.array(torch_tensor.cpu().numpy())


def process_batch(batch):
    """Convert a batch of PyTorch tensors to JAX arrays."""
    return {"src": torch_to_jax(batch[0]), "tgt": torch_to_jax(batch[1])}


def save_results(
    task, ctrl, data, name, ts, directory="results", prefix="sssm", meta=None
):
    path = os.path.join(directory, task, prefix)
    os.makedirs(path, exist_ok=True)

    fname = f"{prefix}-{ctrl}-{name}-{ts}.txt"
    fpath = os.path.join(path, fname)

    with open(fpath, "w") as f:
        if meta:
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
            f.write("\n")

        if isinstance(data, dict):
            for k, v in data.items():
                f.write(f"# {k}\n")
                for item in v:
                    f.write(f"{item}\n")
                f.write("\n")
        else:
            for item in data:
                f.write(f"{item}\n")

    print(f"Data saved to {fpath}")
    return fpath


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Setup distributed training
    device, local_rank, rank, world_size, main_process = setup(FLAGS)

    if main_process:
        colored_print(
            "\nLyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant.",
            Colors.OKBLUE,
        )

    # Prepare directories
    checkpoint_dir = "checkpoints"
    if main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs("results/", exist_ok=True)

    # Hyperparameters
    n_layers = 4
    dropout = 0.10
    num_eigh = 24
    k_y = 2
    k_u = 3
    learnable_m_y = True

    # Data loader hyperparameters
    bsz = 80
    preprocess = True
    shift = 1

    # Data paths
    mujoco_v1_base = f"data/mujoco-v1/{FLAGS.controller}/"
    mujoco_v2_base = f"data/mujoco-v2/{FLAGS.controller}/"
    mujoco_v3_base = f"data/mujoco-v3/{FLAGS.controller}/"

    # Initialize dataset variable
    dataset = None

    # Handle mujoco-v1 and mujoco-v2 tasks
    if FLAGS.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = mujoco_v1_base if FLAGS.task == "mujoco-v1" else mujoco_v2_base
        train_data = {
            "inputs": f"{base_path}/train_inputs.npy",
            "targets": f"{base_path}/train_targets.npy",
        }
        val_data = {
            "inputs": f"{base_path}/val_inputs.npy",
            "targets": f"{base_path}/val_targets.npy",
        }
    elif FLAGS.task == "mujoco-v3":
        # Note: You might need to adjust this part for JAX compatibility
        dataset = jnp.load(f"{mujoco_v3_base}{FLAGS.controller}_ResNet-18.pt")
        train_data, val_data = split_data(dataset)
    else:
        raise ValueError("Invalid task")

    # Data loading
    train_loader = get_dataloader(
        data=train_data,
        task=FLAGS.task,
        bsz=bsz,
        shift=shift,
        preprocess=preprocess,
        shuffle=True,
        distributed=world_size > 1,
        rank=local_rank,
        world_size=world_size,
        device=device,
    )

    val_loader = get_dataloader(
        data=val_data,
        task=FLAGS.task,
        bsz=bsz,
        shift=shift,
        preprocess=preprocess,
        shuffle=False,
        distributed=world_size > 1,
        rank=local_rank,
        world_size=world_size,
        device=device,
    )

    # Get data dimensions
    sample_batch = next(iter(train_loader))
    d_in = sample_batch[0].shape[-1]
    d_out = sample_batch[1].shape[-1]
    sl = sample_batch[0].shape[1]

    def forward_fn(*args, **kwargs):
        return model.Architecture(
            name=None,
            d_model=d_in,
            d_target=d_out,
            num_layers=n_layers,
            dropout=dropout,
            input_len=sl,
            num_eigh=num_eigh,
            auto_reg_k_u=k_u,
            auto_reg_k_y=k_y,
            learnable_m_y=learnable_m_y,
        )(*args, **kwargs)

    forward = hk.transform_with_state(forward_fn)

    # Training hyperparameters
    num_epochs = 3
    steps_per_epoch = len(train_loader)
    num_steps = steps_per_epoch * num_epochs
    warmup_steps = num_steps // 8
    eval_period = num_steps // 16
    patience: int = 10

    if main_process:
        colored_print(f"\nUsing batch size: {bsz}", Colors.OKCYAN)
        colored_print(f"Number of epochs: {num_epochs}", Colors.OKCYAN)
        colored_print(f"Steps per epoch: {steps_per_epoch}", Colors.OKCYAN)
        colored_print(f"=> Number of training steps: {num_steps}", Colors.OKCYAN)

    # Optimizer
    weight_decay = 1e-1
    max_lr = 6e-4

    opt = optimizer.get_optimizer(
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=max_lr,
        weight_decay=weight_decay,
        m_y_learning_rate=5e-5,
        m_y_weight_decay=0,
    )

    # Create experiment
    rng = jax.random.PRNGKey(0)
    exp = experiment.Experiment(forward=forward, optimizer=opt, rng=rng, main_process=main_process)

    # Training loop
    best_val_loss = float("inf")
    best_model_step = 0
    patient_counter = 0
    train_losses = []
    val_losses = []
    val_time_steps = []

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for epoch in range(num_epochs):
        train_iter = iter(train_loader)

        pbar = tqdm.tqdm(range(steps_per_epoch), disable=not main_process)
        for step in pbar:
            global_step = epoch * steps_per_epoch + step

            try:
                torch_inputs = next(train_iter)
            except StopIteration:
                break

            jax_inputs = process_batch(torch_inputs)

            metrics = exp.step(jax_inputs)
            train_loss = metrics["loss"]
            train_losses.append(train_loss)

            if main_process:
                pbar.set_description(
                    f"Step {global_step} - train/loss: {train_loss:.4f}"
                )

            if global_step % eval_period == 0 or global_step == num_steps - 1:
                if main_process:
                    colored_print(
                        f"\nLyla: Evaluating the model at step {global_step}.",
                        Colors.OKCYAN,
                    )

                val_data = [process_batch(batch) for batch in val_loader]
                val_metrics = exp.eval_epoch(val_data)
                val_loss = val_metrics["loss"] / 10
                val_losses.append(val_loss)
                val_time_steps.append(global_step)

                if main_process:
                    colored_print(f"\nValidation Loss: {val_loss:.4f}.", Colors.OKCYAN)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_step = global_step
                        patient_counter = 0

                        # Save checkpoint
                        model_checkpoint = f"sssm-{FLAGS.controller}-model_step-{best_model_step}-{timestamp}.pkl"
                        model_path = os.path.join(checkpoint_dir, model_checkpoint)

                        exp.save_checkpoint(model_path)

                        colored_print(
                            f"Lyla: New best model at step {best_model_step}. "
                            f"Validation loss: {val_loss:.4f}. "
                            f"Model saved as {model_path}",
                            Colors.OKGREEN,
                        )
                    else:
                        patient_counter += 1
                        colored_print(
                            f"Lyla: No improvement for {patient_counter} eval periods. "
                            f"Current best loss: {best_val_loss:.4f}.",
                            Colors.WARNING,
                        )

                if patient_counter >= patience:
                    if main_process:
                        colored_print(
                            f"Lyla: Reached patience limit of {patience}. "
                            f"Stopping training at step {global_step}...",
                            Colors.FAIL,
                        )
                    break

        if patient_counter >= patience:
            break

    # Post-training processing
    if main_process:
        save_results(
            FLAGS.task, FLAGS.controller, train_losses, "train_losses", timestamp
        )
        save_results(FLAGS.task, FLAGS.controller, val_losses, "val_losses", timestamp)
        save_results(
            FLAGS.task, FLAGS.controller, val_time_steps, "val_time_steps", timestamp
        )

        colored_print(
            f"Lyla: Training complete! Best model was at step {best_model_step} with validation loss {best_val_loss:.4f}.",
            Colors.OKGREEN,
        )

    cleanup()


if __name__ == "__main__":
    app.run(main)
