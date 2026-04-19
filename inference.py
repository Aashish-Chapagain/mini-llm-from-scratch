# pyright: basic
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

   
    if top_k > 0 and top_k < probs.shape[0]:
        keep = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)

   
    if 0.0 < top_p < 1.0:
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


def _normalize_text(text):
    return str(text).strip().lower()


def build_chat_prompt(history, user_message, max_context_turns=6):
    """Build a chat prompt using recent conversation turns and the current user input."""
    clipped_history = history[-(max_context_turns * 2) :] if max_context_turns > 0 else history
    parts = []
    for turn in clipped_history:
        role = turn.get("role", "").strip().lower()
        text = _normalize_text(turn.get("text", ""))
        if not text:
            continue
        if role == "user":
            parts.append(f"<user> {text} ")
        elif role == "bot":
            parts.append(f"<bot> {text} ")

    current_user = _normalize_text(user_message)
    parts.append(f"<user> {current_user} <bot> ")
    return "".join(parts)


def text_to_ids(text, char_to_id, pad_id):
    return [char_to_id.get(char, pad_id) for char in text]


def generate_chat_reply(
    model,
    char_to_id,
    id_to_char,
    history,
    user_message,
    pad_id,
    max_context_turns=6,
    max_len=120,
    temperature=0.65,
    top_k=12,
    top_p=0.9,
    repetition_penalty=1.15,
):
    prompt = build_chat_prompt(history, user_message, max_context_turns=max_context_turns)
    input_ids = text_to_ids(prompt, char_to_id, pad_id)
    generated_ids = generate(
        model,
        input_ids,
        pad_id=pad_id,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    full_generated = convert_ids_to_text(generated_ids, id_to_char, pad_id=pad_id)
    return extract_bot_reply(full_generated)


def load_inference_components(vocab_path="vocab.json", weights_path="model_weights.npz"):
    char_to_id, id_to_char = load_vocab(vocab_path)
    pad_id = char_to_id.get("<PAD>", 0)
    model = build_model_from_checkpoint(weights_path)
    return model, char_to_id, id_to_char, pad_id


def main():
    model, char_to_id, id_to_char, pad_id = load_inference_components(
        vocab_path="vocab.json",
        weights_path="model_weights.npz",
    )

    history = []
    print("Chat started. Type 'quit' to exit.")
    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            print("Chat ended.")
            break

        bot_reply = generate_chat_reply(
            model=model,
            char_to_id=char_to_id,
            id_to_char=id_to_char,
            history=history,
            user_message=user_prompt,
            pad_id=pad_id,
            max_context_turns=6,
            max_len=120,
            temperature=0.65,
            top_k=12,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        print("Bot:", bot_reply)

        history.append({"role": "user", "text": user_prompt})
        history.append({"role": "bot", "text": bot_reply})


if __name__ == "__main__":
    main()