import json
import numpy as np




def load_dataset(file_path = "dataset.json"):
    with open(file_path, "r", encoding="utf-8") as f: 
        return json.load(f)




def build_vocab(dataset):
    vocab = set()
    for pair in dataset:
        vocab.update(pair["input"].lower())
        vocab.update(pair["output"].lower())
    
    vocab = sorted(vocab)

    char_to_id = {char: idx for idx, char in enumerate(vocab)}
    id_to_char = {idx: char for idx, char in enumerate(vocab)}
    return char_to_id, id_to_char
    



def encoding(text, char_to_id):
    return [char_to_id[char] for char in text if char in char_to_id]




def pad_sequence(seq, max_length, pad_id):
    return seq + [pad_id] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length]



def tokenize_dataset(dataset, char2idx,  max_length):
    inputs = []
    targets = []


    for pair in dataset:
        text = "<user> " + pair["input"] + " <bot> " + pair["output"]
        text = text.lower()
        encoded = encoding(text, char2idx)
        padded = pad_sequence(encoded, max_length, pad_id = char2idx.get("<PAD>", 0)) 

        inputs.append(padded[:-1])
        targets.append(padded[1:])

    return np.array(inputs), np.array(targets)



def save_vocab(char_to_id, id_to_char, file_path = "vocab.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"char_to_id": char_to_id, "id_to_char": id_to_char}, f, indent=2, ensure_ascii=False)
    


def main():
    dataset = load_dataset()
    char_to_id, id_to_char = build_vocab(dataset)
    save_vocab(char_to_id, id_to_char)

    max_length = 256
    inputs, targets = tokenize_dataset(dataset, char_to_id, max_length)

    np.save("inputs.npy", inputs)
    np.save("targets.npy", targets)
    print(f"Tokenization complete. Vocabulary size: {len(char_to_id)}. Dataset size: {len(dataset)}.{inputs.shape} input-target pairs saved.")


main()