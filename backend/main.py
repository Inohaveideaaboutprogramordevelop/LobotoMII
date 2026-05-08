import io
import os
import sys
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(__file__))
from LobotoMII import TomodachiTTS, PRESETS, FORMATOS

app = FastAPI(title="LobotoMII API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

MIME_TYPES = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "ogg":  "audio/ogg",
    "opus": "audio/ogg; codecs=opus",
    "m4a":  "audio/mp4",
    "flac": "audio/flac",
}


class SynthRequest(BaseModel):
    texto:        str           = Field(..., min_length=1, max_length=500)
    preset:       str           = "default"
    formato:      str           = "wav"
    semitonos:    Optional[float] = None
    espeak_speed: Optional[int]   = None
    chorus:       Optional[float] = None
    vibrato_amp:  Optional[float] = None
    brillo:       Optional[float] = None
    volumen:      Optional[float] = None


def _synthesize(req: SynthRequest) -> bytes:
    if req.preset not in PRESETS:
        raise ValueError(f"Preset '{req.preset}' no válido")
    if req.formato not in FORMATOS:
        raise ValueError(f"Formato '{req.formato}' no válido")

    cfg = PRESETS[req.preset].copy()
    for attr, key in [
        ("semitonos",    "semitonos"),
        ("espeak_speed", "espeak_speed"),
        ("chorus",       "chorus"),
        ("vibrato_amp",  "vibrato_amp"),
        ("brillo",       "brillo"),
        ("volumen",      "volumen"),
    ]:
        val = getattr(req, attr)
        if val is not None:
            cfg[key] = int(val) if key == "espeak_speed" else val

    tts = TomodachiTTS(**cfg)

    if req.formato == "wav":
        audio, sr = tts.sintetizar(req.texto)
        data16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, sr, data16)
        return buf.getvalue()

    with tempfile.NamedTemporaryFile(suffix=f".{req.formato}", delete=False) as f:
        tmp = f.name
    try:
        tts.guardar(req.texto, tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


@app.get("/")
async def root():
    return {"status": "ok", "service": "LobotoMII API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/presets")
async def get_presets():
    return {name: cfg["descripcion"] for name, cfg in PRESETS.items()}


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(executor, _synthesize, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    mime = MIME_TYPES.get(req.formato, "audio/wav")
    return Response(
        content=data,
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="mii_{req.preset}.{req.formato}"',
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
