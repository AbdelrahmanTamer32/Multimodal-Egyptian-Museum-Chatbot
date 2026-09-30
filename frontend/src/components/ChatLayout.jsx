import React, { useRef, useState } from "react";

export default function ChatLayout() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Welcome to the Interactive Museum Guide. You can type a question, use your microphone, or upload an artifact image."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const audioInputRef = useRef(null);
  const imageInputRef = useRef(null);

  const sendTextQuestion = async () => {
    const question = input.trim();
    if (!question || loading) return;

    const newUserMessage = { role: "user", text: question };
    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat/text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question })
      });

      if (!response.ok) {
        throw new Error("Failed to get response from backend.");
      }

      const data = await response.json();

      const assistantMessage = {
        role: "assistant",
        text: data.answer || "No answer returned."
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Error: could not connect to the text backend."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendAudioQuestion = async (file) => {
    if (!file || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        text: `🎤 Audio question: ${file.name}`
      }
    ]);

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/chat/audio", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Failed to get audio response from backend.");
      }

      const data = await response.json();

      const assistantMessage = {
        role: "assistant",
        text: data.answer || "No answer returned from audio endpoint."
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Error: could not connect to the audio backend."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendImageQuestion = async (file) => {
    if (!file || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        text: `🖼 Image question: ${file.name}`
      }
    ]);

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", "");
      formData.append("lang", "en");

      const response = await fetch("http://127.0.0.1:8000/chat/image", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Failed to get image response from backend.");
      }

      const data = await response.json();

      const assistantMessage = {
        role: "assistant",
        text: data.answer || "No answer returned from image endpoint."
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Error: could not connect to the image backend."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleAudioFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await sendAudioQuestion(file);
    }
    e.target.value = "";
  };

  const handleImageFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await sendImageQuestion(file);
    }
    e.target.value = "";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendTextQuestion();
    }
  };

  return (
    <div className="app-shell">
      <div className="app-container">
        <aside className="sidebar">
          <h1 className="sidebar-title">Museum Guide</h1>
          <button
            className="new-chat-btn"
            onClick={() =>
              setMessages([
                {
                  role: "assistant",
                  text: "Welcome to the Interactive Museum Guide. You can type a question, use your microphone, or upload an artifact image."
                }
              ])
            }
          >
            New Chat
          </button>

          <div className="chat-history">
            <div className="history-item">Museum conversation</div>
          </div>

          <div className="sidebar-footer">Simple prototype for user testing</div>
        </aside>

        <main className="main-panel">
          <header className="topbar">
            <div>
              <h2>Interactive Museum Guide</h2>
              <p>Ask by text, voice, or image</p>
            </div>
            <div className="prototype-badge">Prototype UI</div>
          </header>

          <section className="chat-area">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message-row ${message.role === "user" ? "user-row" : "assistant-row"}`}
              >
                <div className={`message-bubble ${message.role}`}>
                  {message.text}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row assistant-row">
                <div className="message-bubble assistant">Thinking...</div>
              </div>
            )}
          </section>

          <footer className="input-area">
            <div className="input-box">
              <textarea
                placeholder="Ask about an artifact..."
                rows="3"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />

              <input
                ref={audioInputRef}
                type="file"
                accept=".mp3,.wav,.m4a,.webm,.ogg"
                style={{ display: "none" }}
                onChange={handleAudioFileChange}
              />

              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={handleImageFileChange}
              />

              <div className="input-actions">
                <div className="left-actions">
                  <button
                    type="button"
                    onClick={() => audioInputRef.current?.click()}
                  >
                    🎤 Voice
                  </button>

                  <button
                    type="button"
                    onClick={() => imageInputRef.current?.click()}
                  >
                    🖼 Upload Image
                  </button>
                </div>

                <button className="send-btn" onClick={sendTextQuestion} disabled={loading}>
                  {loading ? "Sending..." : "Send"}
                </button>
              </div>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}