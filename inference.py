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


def sample_with_top_k_top_p(logits, top_k=20, top_p=0.9, temperature=0.7):
    scaled = logits / max(temperature, 1e-6)
    probs = softmax(scaled)

    # Top-k filtering first.
    if top_k is not None and top_k > 0 and top_k < probs.shape[0]:
        keep = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)

    # Then nucleus (top-p) filtering.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)

        nucleus_mask = cumsum <= top_p
        if np.any(nucleus_mask):
            first_false = np.argmax(~nucleus_mask) if np.any(~nucleus_mask) else len(nucleus_mask)
            if first_false < len(nucleus_mask):
                nucleus_mask[first_false] = True
        else:
            nucleus_mask[0] = True

        keep_sorted = sorted_idx[nucleus_mask]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep_sorted] = True
        probs = np.where(mask, probs, 0.0)

    prob_sum = np.sum(probs)
    if prob_sum <= 0.0:
        return int(np.argmax(scaled))

    probs /= prob_sum
    return int(np.random.choice(len(probs), p=probs))


def generate(
    model,
    input_ids,
    pad_id,
    max_len=100,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.1,
):
    generated = list(input_ids)
    if len(generated) == 0:
        generated = [pad_id]

    for _ in range(max_len):
        # Keep only the latest context that fits positional encoding length.
        context = np.array(generated[-model.max_seq_len :], dtype=np.int64)[None, :]
        logits = model.forward(context)
        next_token_logits = logits[0, -1].copy()

        
        next_token_logits[pad_id] = -1e9

        
        if repetition_penalty > 1.0:
            for token_id in set(generated[-64:]):
                next_token_logits[token_id] /= repetition_penalty

        next_token_id = sample_with_top_k_top_p(
            next_token_logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        generated.append(next_token_id)

    return generated


def convert_ids_to_text(ids, id_to_char, pad_id):
    clean_ids = [token_id for token_id in ids if token_id != pad_id]
    return "".join(id_to_char.get(token_id, "") for token_id in clean_ids)


def extract_bot_reply(full_text):
    marker = "<bot>"
    idx = full_text.rfind(marker)
    if idx == -1:
        return full_text.strip()

    reply = full_text[idx + len(marker) :]

    
    next_user = reply.find("<user>")
    next_bot = reply.find("<bot>")
    cut_points = [p for p in [next_user, next_bot] if p != -1]
    if cut_points:
        reply = reply[: min(cut_points)]

    return reply.strip()


def main():
    char_to_id, id_to_char = load_vocab("vocab.json")
    pad_id = char_to_id.get("<PAD>", 0)
    model = build_model_from_checkpoint("model_weights.npz")

    user_prompt = "What would you name your boat if you had one?"
    prompt = f"<user> {user_prompt.lower()} <bot> "
    input_ids = [char_to_id.get(char, pad_id) for char in prompt]
    generated_ids = generate(
        model,
        input_ids,
        pad_id=pad_id,
        max_len=120,
        temperature=0.65,
        top_k=12,
        top_p=0.9,
        repetition_penalty=1.15,
    )
    full_generated = convert_ids_to_text(generated_ids, id_to_char, pad_id=pad_id)
    bot_reply = extract_bot_reply(full_generated)
    print("Prompt:", user_prompt)
    print("Generated reply:", bot_reply)


if __name__ == "__main__":
    main()