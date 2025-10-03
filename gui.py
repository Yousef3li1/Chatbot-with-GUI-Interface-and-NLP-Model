import os
import tkinter as tk
from tkinter import scrolledtext, END

from src.inference import ChatbotInference


MODEL_PATH = "model.h5"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"
INTENTS_PATH = os.path.join("data", "intents.json")


class ChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NLP Chatbot")
        self.root.geometry("600x500")

        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state="disabled")
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry = tk.Entry(self.root)
        self.entry.pack(padx=10, pady=5, fill=tk.X)
        self.entry.bind("<Return>", self.on_send)

        self.send_btn = tk.Button(self.root, text="Send", command=self.on_send)
        self.send_btn.pack(padx=10, pady=5)

        self.bot = ChatbotInference(MODEL_PATH, WORDS_PATH, CLASSES_PATH, INTENTS_PATH)
        self._append("Bot", "Hello! I'm ready to chat.")

    def _append(self, speaker: str, message: str) -> None:
        self.chat_area.configure(state="normal")
        self.chat_area.insert(END, f"{speaker}: {message}\n")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(END)

    def on_send(self, event=None) -> None:  # type: ignore[no-untyped-def]
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, END)
        self._append("You", text)
        response = self.bot.get_response(text)
        self._append("Bot", response)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

