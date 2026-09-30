from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pipeline_image import ImageChatbot
from src.pipeline_text import TextChatbot
from src.pipeline_audio import AudioChatbot

app = FastAPI(title="Museum Guide Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
text_bot = TextChatbot(project_root=PROJECT_ROOT)
audio_bot = AudioChatbot(project_root=PROJECT_ROOT)
image_bot = ImageChatbot(project_root=PROJECT_ROOT)

class TextRequest(BaseModel):
    question: str

@app.post("/chat/text")
def chat_text(req: TextRequest):
    out = text_bot.ask(req.question)
    return out

@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    # save uploaded audio temporarily
    temp_dir = PROJECT_ROOT / "outputs" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    content = await file.read()
    temp_path.write_bytes(content)

    out = audio_bot.ask_audio(temp_path)

    return out

@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    question: str = "",
    lang: str = "en",
):
    # Save uploaded image temporarily
    temp_dir = PROJECT_ROOT / "outputs" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # keep extension if possible
    filename = file.filename or "upload.jpg"
    temp_path = temp_dir / filename

    content = await file.read()
    temp_path.write_bytes(content)

    out = image_bot.ask_image(temp_path, question=question, lang=lang)
    return out