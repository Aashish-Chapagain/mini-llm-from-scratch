# 🧠 Mini LLM From Scratch

A minimal implementation of a **decoder-only Transformer (GPT-style)** built completely from scratch using PyTorch.

This project focuses on understanding the **core building blocks of Large Language Models (LLMs)** — from tokenization to training and generation.

---

## 🚀 Project Goal

The goal of this project is **learning by building**, not just using APIs.

Instead of relying on pre-trained models, this project explores:

- How language models actually learn
- Why pretraining is necessary
- How fine-tuning changes behavior

---

## ⚠️ Key Learning (Important Insight)

Initially, I tried training a chatbot directly on a small Q&A dataset (~274 samples).

👉 Result: Poor and meaningless outputs.

### 🔍 Problem
- The model had **no understanding of language**
- It was trying to "answer" without knowing grammar or structure


