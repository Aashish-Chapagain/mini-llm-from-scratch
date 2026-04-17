import os
import threading
import uuid
from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for
from inference import generate_chat_reply, load_inference_components


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

model, char_to_id, id_to_char, pad_id = load_inference_components(
    vocab_path="vocab.json",
    weights_path="model_weights.npz",
)

chat_memory = {}
memory_lock = threading.Lock()


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>miniBrain Chat</title>
  <style>
    :root {
      --bg: #f4f2eb;
      --panel: #fffaf0;
      --ink: #1e1b18;
      --accent: #1f6f8b;
      --line: #c8c2b4;
      --muted: #655f54;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: radial-gradient(circle at top right, #e2ded2 0%, var(--bg) 45%, #efe8d9 100%);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }

    .shell {
      width: min(900px, 100%);
      background: var(--panel);
      border: 2px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.1);
      overflow: hidden;
    }

    .head {
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: linear-gradient(90deg, #efe6d2, #f8f2e3);
    }

    .title {
      margin: 0;
      letter-spacing: 0.04em;
      font-size: 1.05rem;
      text-transform: uppercase;
    }

    .chat {
      height: 55vh;
      overflow-y: auto;
      padding: 16px;
      display: grid;
      gap: 10px;
      align-content: start;
    }

    .msg {
      max-width: 85%;
      padding: 10px 12px;
      border-radius: 12px;
      line-height: 1.35;
      white-space: pre-wrap;
      animation: fade 220ms ease;
    }

    .user {
      justify-self: end;
      background: #d8edf5;
      border: 1px solid #9bc7d8;
    }

    .bot {
      justify-self: start;
      background: #f2ead7;
      border: 1px solid #d6c8a9;
    }

    .controls {
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      padding: 12px;
      background: #fbf6ea;
    }

    input[type="text"] {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      font-size: 1rem;
      font-family: inherit;
    }

    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 600;
      font-family: inherit;
    }

    .send {
      background: var(--accent);
      color: white;
    }

    .clear {
      background: #ddd4bf;
      color: var(--ink);
    }

    .hint {
      font-size: 0.85rem;
      color: var(--muted);
      padding: 0 14px 12px;
    }

    @keyframes fade {
      from { opacity: 0; transform: translateY(3px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 720px) {
      .chat { height: 62vh; }
      .controls { grid-template-columns: 1fr; }
      .msg { max-width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="head">
      <h1 class="title">miniBrain Chat</h1>
      <span>Session memory enabled</span>
    </header>

    <section id="chat" class="chat">
      {% if history %}
        {% for turn in history %}
          <article class="msg {{ turn['role'] }}">{{ turn['role']|capitalize }}: {{ turn['text'] }}</article>
        {% endfor %}
      {% else %}
        <article class="msg bot">Bot: Ask me anything to start the conversation.</article>
      {% endif %}
    </section>

    <form id="chat-form" class="controls">
      <input id="message" type="text" name="message" placeholder="Type your message..." autocomplete="off" required>
      <button class="send" type="submit">Send</button>
      <button class="clear" type="button" id="clear-btn">Clear Memory</button>
    </form>
    <div class="hint">Memory is kept per browser session and used as context for each new reply.</div>
  </main>

  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("chat-form");
    const messageInput = document.getElementById("message");
    const clearBtn = document.getElementById("clear-btn");

    const addMessage = (role, text) => {
      const item = document.createElement("article");
      item.className = `msg ${role}`;
      item.textContent = `${role.charAt(0).toUpperCase() + role.slice(1)}: ${text}`;
      chat.appendChild(item);
      chat.scrollTop = chat.scrollHeight;
    };

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const userText = messageInput.value.trim();
      if (!userText) return;

      addMessage("user", userText);
      messageInput.value = "";

      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });
      const data = await response.json();
      addMessage("bot", data.reply || "I could not generate a reply.");
    });

    clearBtn.addEventListener("click", async () => {
      await fetch("/clear", { method: "POST" });
      chat.innerHTML = "";
      addMessage("bot", "Memory cleared for this session.");
    });
  </script>
</body>
</html>
"""


def _session_id():
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    return sid


def _get_history():
    sid = _session_id()
    with memory_lock:
        return chat_memory.setdefault(sid, [])


@app.route("/", methods=["GET"])
def index():
    history = _get_history()
    return render_template_string(HTML_PAGE, history=history)


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    history = _get_history()
    bot_reply = generate_chat_reply(
        model=model,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
        history=history,
        user_message=user_message,
        pad_id=pad_id,
        max_context_turns=6,
        max_len=120,
        temperature=0.65,
        top_k=12,
        top_p=0.9,
        repetition_penalty=1.15,
    )

    with memory_lock:
        history.append({"role": "user", "text": user_message})
        history.append({"role": "bot", "text": bot_reply})

    return jsonify({"reply": bot_reply, "history": history})


@app.route("/clear", methods=["POST"])
def clear_memory():
    sid = _session_id()
    with memory_lock:
        chat_memory[sid] = []
    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
