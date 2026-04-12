import json
import numpy as np
from model import FullModel


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def load_vocab(file_path="vocab.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    char_to_id = vocab["char_to_id"]
    id_to_char = {int(k): v for k, v in vocab["id_to_char"].items()}
    return char_to_id, id_to_char


def build_model_from_checkpoint(weights_path="model_weights.npz"):
    state = np.load(weights_path)
    meta = state["meta"].astype(np.int64)
    model = FullModel(
        vocab_size=int(meta[0]),
        d_model=int(meta[1]),
        num_heads=int(meta[2]),
        d_ff=int(meta[3]),
        num_layers=int(meta[4]),
        max_seq_len=int(meta[5]),
    )
    model.load_weights(weights_path)
    return model


def generate(model, input_ids, max_len=100, temperature=1.0):
    generated = list(input_ids)

    for _ in range(max_len):
        # Keep only the latest context that fits positional encoding length.
        context = np.array(generated[-model.max_seq_len :], dtype=np.int64)[None, :]
        logits = model.forward(context)
        next_token_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = softmax(next_token_logits)
        next_token_id = int(np.random.choice(len(probs), p=probs))
        generated.append(next_token_id)

    return generated


def convert_ids_to_text(ids, id_to_char):
    return "".join(id_to_char.get(token_id, "") for token_id in ids)


def main():
    char_to_id, id_to_char = load_vocab("vocab.json")
    model = build_model_from_checkpoint("model_weights.npz")

    prompt = "hello, how are you? "
    input_ids = [char_to_id.get(char, 0) for char in prompt.lower()]
    generated_ids = generate(model, input_ids, max_len=100, temperature=1.0)
    generated_text = convert_ids_to_text(generated_ids, id_to_char)
    print("Generated text:", generated_text)


if __name__ == "__main__":
    main()