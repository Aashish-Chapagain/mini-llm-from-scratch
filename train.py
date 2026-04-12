import numpy as np 
from model import FullModel, SGDOptimizer



inputs = np.load("inputs.npy")
targets = np.load("targets.npy")    



def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets):
    probs = softmax(logits)
    B, T, V = probs.shape
    probs_flat = probs.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    correct_probs = probs_flat[np.arange(B * T), targets_flat]
    loss = -np.mean(np.log(correct_probs + 1e-9))
    return loss, probs


def back_propagation(model, targets, probs):
    B, T, V = probs.shape

    
    dlogits = probs.copy()
    batch_indices = np.arange(B)[:, None]
    time_indices = np.arange(T)[None, :]
    dlogits[batch_indices, time_indices, targets] -= 1.0
    dlogits /= (B * T)
    model.backward(dlogits)


def train(model, inputs, targets, batch_size = 16, epochs=150, learning_rate=1e-4):
    optimizer = SGDOptimizer(learning_rate=learning_rate)
    num_samples = inputs.shape[0]

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            x_batch = inputs[batch_indices]
            y_batch = targets[batch_indices]

            logits = model.forward(x_batch)
            loss, probs = cross_entropy_loss(logits, y_batch)
            back_propagation(model, y_batch, probs)
            optimizer.step(model.parameters_and_grads())

            epoch_loss += loss
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {avg_epoch_loss:.6f} | Batch size: {batch_size}")


def main():
    vocab_size = int(np.max(np.concatenate([inputs.reshape(-1), targets.reshape(-1)])) + 1)
    max_seq_len = inputs.shape[1]

    model = FullModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_seq_len=max_seq_len,
    )

    train(model, inputs, targets, batch_size=16, epochs=150, learning_rate=1e-4)
    model.save_weights("model_weights.npz")
    model.save_block_checkpoints("block_checkpoints.npz")
    print("Saved checkpoint: model_weights.npz")


if __name__ == "__main__":
    main()
