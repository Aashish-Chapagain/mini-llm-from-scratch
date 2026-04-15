import os
import numpy as np 
from model import FullModel, AdamOptimizer




inputs = np.load("inputs.npy")
targets = np.load("targets.npy")    



def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets, label_smoothing=0.0):
    probs = softmax(logits)
    B, T, V = probs.shape

    if label_smoothing <= 0.0:
        probs_flat = probs.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        correct_probs = probs_flat[np.arange(B * T), targets_flat]
        loss = -np.mean(np.log(correct_probs + 1e-9))
    else:
        target_dist = np.full((B, T, V), label_smoothing / V, dtype=probs.dtype)
        batch_indices = np.arange(B)[:, None]
        time_indices = np.arange(T)[None, :]
        target_dist[batch_indices, time_indices, targets] += (1.0 - label_smoothing)
        loss = -np.mean(np.sum(target_dist * np.log(probs + 1e-9), axis=-1))

    return loss, probs


def back_propagation(model, targets, probs, label_smoothing=0.0):
    B, T, V = probs.shape

    dlogits = probs.copy()
    if label_smoothing <= 0.0:
        batch_indices = np.arange(B)[:, None]
        time_indices = np.arange(T)[None, :]
        dlogits[batch_indices, time_indices, targets] -= 1.0
    else:
        target_dist = np.full((B, T, V), label_smoothing / V, dtype=probs.dtype)
        batch_indices = np.arange(B)[:, None]
        time_indices = np.arange(T)[None, :]
        target_dist[batch_indices, time_indices, targets] += (1.0 - label_smoothing)
        dlogits -= target_dist

    dlogits /= (B * T)
    model.backward(dlogits)


def evaluate_loss(model, x_data, y_data, batch_size=32):
    total_loss = 0.0
    num_batches = 0

    for start in range(0, x_data.shape[0], batch_size):
        x_batch = x_data[start:start + batch_size]
        y_batch = y_data[start:start + batch_size]
        logits = model.forward(x_batch)
        loss, _ = cross_entropy_loss(logits, y_batch)
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train(
    model,
    train_inputs,
    train_targets,
    val_inputs,
    val_targets,
    batch_size=16,
    max_epochs=800,
    learning_rate=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    patience=40,
    min_delta=1e-4,
):
    optimizer = AdamOptimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=1.0,
    )
    num_samples = train_inputs.shape[0]
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            x_batch = train_inputs[batch_indices]
            y_batch = train_targets[batch_indices]

            logits = model.forward(x_batch)
            loss, probs = cross_entropy_loss(logits, y_batch, label_smoothing=label_smoothing)
            back_propagation(model, y_batch, probs, label_smoothing=label_smoothing)
            optimizer.step(model.parameters_and_grads())

            epoch_loss += loss
            num_batches += 1

        train_loss = epoch_loss / max(num_batches, 1)
        val_loss = evaluate_loss(model, val_inputs, val_targets, batch_size=batch_size)

        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_weights("best_model_weights.npz")
            model.save_block_checkpoints("best_block_checkpoints.npz")
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best Val: {best_val_loss:.6f} | No Improve: {epochs_without_improvement}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")
            break

    # Restore best weights before returning.
    model.load_weights("best_model_weights.npz")


def main():
    vocab_size = int(np.max(np.concatenate([inputs.reshape(-1), targets.reshape(-1)])) + 1)
    max_seq_len = inputs.shape[1]

    # 90/10 split for validation.
    num_samples = inputs.shape[0]
    indices = np.random.permutation(num_samples)
    split = max(1, int(0.1 * num_samples))
    val_idx = indices[:split]
    train_idx = indices[split:]

    train_inputs = inputs[train_idx]
    train_targets = targets[train_idx]
    val_inputs = inputs[val_idx]
    val_targets = targets[val_idx]

    model = FullModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_seq_len=max_seq_len,
    )

    # Resume from latest checkpoint if available.
    if os.path.exists("model_weights.npz"):
        try:
            model.load_weights("model_weights.npz")
            print("Loaded checkpoint: model_weights.npz (resuming training)")
        except ValueError as exc:
            print(f"Checkpoint skipped due to mismatch: {exc}")

    train(
        model,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        batch_size=16,
        max_epochs=800,
        learning_rate=3e-4,
        weight_decay=1e-4,
        label_smoothing=0.05,
        patience=40,
        min_delta=1e-4,
    )
    model.save_weights("model_weights.npz")
    model.save_block_checkpoints("block_checkpoints.npz")
    print("Saved checkpoint: model_weights.npz")
    


if __name__ == "__main__":
    main()
