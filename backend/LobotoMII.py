"""
╔══════════════════════════════════════════════════════════════════════╗
║          TOMODACHI TTS — Text-to-Speech estilo Tomodachi Life        ║
║              Compatible con Windows, Mac y Linux                     ║
╚══════════════════════════════════════════════════════════════════════╝

Requisitos:
    1. eSpeak-NG:
       Windows → https://github.com/espeak-ng/espeak-ng/releases
                 Descargar el .msi e instalar (marcar "Add to PATH")
       Mac     → brew install espeak-ng
       Linux   → sudo apt install espeak-ng

    2. FFmpeg (para formatos mp3, ogg, opus, m4a):
       Windows → https://www.gyan.dev/ffmpeg/builds/  (ffmpeg-release-essentials.zip)
                 Extraer y agregar la carpeta /bin al PATH
       Mac     → brew install ffmpeg
       Linux   → sudo apt install ffmpeg

    3. Python:
       pip install numpy scipy librosa soundfile sounddevice pydub

Formatos soportados:
    wav   → universal, sin pérdida
    mp3   → el más compatible (requiere ffmpeg)
    ogg   → open source, buena calidad (requiere ffmpeg)
    opus  → WhatsApp, Telegram, Discord (requiere ffmpeg)
    m4a   → iPhone / Apple (requiere ffmpeg)
    flac  → sin pérdida comprimido (requiere ffmpeg)

Uso:
    python tomodachi_tts.py "Hola! Me llamo Mii!"
    python tomodachi_tts.py "Texto" --preset kawaii
    python tomodachi_tts.py "Texto" --guardar voz.wav
    python tomodachi_tts.py "Texto" --guardar voz.mp3
    python tomodachi_tts.py "Texto" --guardar voz.opus        ← WhatsApp
    python tomodachi_tts.py "Texto" --semitonos 9 --chorus 0.3
    python tomodachi_tts.py --listar-presets
"""

import numpy as np
import subprocess
import tempfile
import argparse
import sys
import os
import shutil

# Intentar importar librosa al inicio para fallar rápido si no está presente
try:
    import librosa
except ImportError:
    print("❌ Error: La librería 'librosa' es necesaria.")
    print("   Instálala con: pip install librosa")
    sys.exit(1)

from scipy import signal
from scipy.io import wavfile


# ══════════════════════════════════════════════════════════════
#  FORMATOS SOPORTADOS
# ══════════════════════════════════════════════════════════════
FORMATOS = {
    "wav":  {"desc": "WAV  — sin pérdida, universal",          "requiere_ffmpeg": False},
    "mp3":  {"desc": "MP3  — el más compatible",               "requiere_ffmpeg": True},
    "ogg":  {"desc": "OGG  — open source, buena calidad",      "requiere_ffmpeg": True},
    "opus": {"desc": "OPUS — WhatsApp / Telegram / Discord",   "requiere_ffmpeg": True},
    "m4a":  {"desc": "M4A  — iPhone / Apple",                  "requiere_ffmpeg": True},
    "flac": {"desc": "FLAC — sin pérdida comprimido",          "requiere_ffmpeg": True},
}


# ══════════════════════════════════════════════════════════════
#  DETECCIÓN AUTOMÁTICA DE HERRAMIENTAS
# ══════════════════════════════════════════════════════════════
def encontrar_espeak() -> str:
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    rutas_windows = [
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
        r"C:\eSpeak NG\espeak-ng.exe",
    ]
    for ruta in rutas_windows:
        if os.path.isfile(ruta):
            return ruta
    raise FileNotFoundError(
        "\n❌ No se encontró eSpeak-NG.\n"
        "   Windows: https://github.com/espeak-ng/espeak-ng/releases\n"
        "   ⚠  Marcar 'Add to PATH' durante la instalación y reiniciar VS Code.\n"
    )


def encontrar_ffmpeg() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    rutas_windows = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for ruta in rutas_windows:
        if os.path.isfile(ruta):
            return ruta
    return None


def encontrar_voz_es(espeak_bin: str) -> str:
    try:
        # FIX Windows: usar stdout=PIPE con encoding explícito en vez de capture_output+text
        proc = subprocess.Popen(
            [espeak_bin, "--voices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_bytes, _ = proc.communicate(timeout=10)
        voces = stdout_bytes.decode("utf-8", errors="ignore")
        for candidato in ["es-419", "es-la", "es-mx", "es", "en"]:
            if candidato in voces:
                return candidato
    except Exception:
        pass
    return "es"


# ══════════════════════════════════════════════════════════════
#  PRESETS
# ══════════════════════════════════════════════════════════════
PRESETS = {
    "default": {
        "descripcion":   "Mii neutro — la voz clásica de Tomodachi Life",
        "espeak_speed":  130,
        "espeak_pitch":  50,
        "semitonos":     7.0,
        "chorus":        0.22,
        "chorus_detune": 0.18,
        "vibrato_hz":    5.5,
        "vibrato_amp":   0.012,
        "brillo":        0.35,
        "volumen":       0.88,
    },
    "kawaii": {
        "descripcion":   "Mii kawaii — aguda, alegre y expresiva",
        "espeak_speed":  150,
        "espeak_pitch":  70,
        "semitonos":     10.0,
        "chorus":        0.28,
        "chorus_detune": 0.20,
        "vibrato_hz":    6.5,
        "vibrato_amp":   0.018,
        "brillo":        0.50,
        "volumen":       0.88,
    },
    "grave": {
        "descripcion":   "Mii adulto — grave, serio, pausado",
        "espeak_speed":  115,
        "espeak_pitch":  30,
        "semitonos":     3.0,
        "chorus":        0.18,
        "chorus_detune": 0.15,
        "vibrato_hz":    4.0,
        "vibrato_amp":   0.008,
        "brillo":        0.20,
        "volumen":       0.88,
    },
    "nino": {
        "descripcion":   "Mii infantil — muy agudo y travieso",
        "espeak_speed":  170,
        "espeak_pitch":  80,
        "semitonos":     12.0,
        "chorus":        0.30,
        "chorus_detune": 0.22,
        "vibrato_hz":    7.0,
        "vibrato_amp":   0.022,
        "brillo":        0.55,
        "volumen":       0.85,
    },
    "robot": {
        "descripcion":   "Mii robot — sintético con personalidad",
        "espeak_speed":  125,
        "espeak_pitch":  45,
        "semitonos":     5.0,
        "chorus":        0.05,
        "chorus_detune": 0.08,
        "vibrato_hz":    0.0,
        "vibrato_amp":   0.0,
        "brillo":        0.10,
        "volumen":       0.85,
    },
    "susurro": {
        "descripcion":   "Mii misterioso — suave y bajo",
        "espeak_speed":  105,
        "espeak_pitch":  42,
        "semitonos":     5.0,
        "chorus":        0.15,
        "chorus_detune": 0.12,
        "vibrato_hz":    2.5,
        "vibrato_amp":   0.008,
        "brillo":        0.08,
        "volumen":       0.55,
    },
    "anciano": {
        "descripcion":   "Mii anciano — ronco y lento",
        "espeak_speed":  100,
        "espeak_pitch":  20,
        "semitonos":     2.0,
        "chorus":        0.25,
        "chorus_detune": 0.25,
        "vibrato_hz":    4.5,
        "vibrato_amp":   0.020,
        "brillo":        0.15,
        "volumen":       0.82,
    },
    "deportivo": {
        "descripcion":   "Mii deportivo — enérgico y directo",
        "espeak_speed":  160,
        "espeak_pitch":  55,
        "semitonos":     6.0,
        "chorus":        0.20,
        "chorus_detune": 0.15,
        "vibrato_hz":    5.0,
        "vibrato_amp":   0.010,
        "brillo":        0.45,
        "volumen":       0.95,
    },
    "timido": {
        "descripcion":   "Mii tímido — suave, dulce y pausado",
        "espeak_speed":  110,
        "espeak_pitch":  60,
        "semitonos":     8.0,
        "chorus":        0.18,
        "chorus_detune": 0.14,
        "vibrato_hz":    4.5,
        "vibrato_amp":   0.014,
        "brillo":        0.22,
        "volumen":       0.60,
    },
    "gruñon": {
        "descripcion":   "Mii gruñón — brusco y malhumorado",
        "espeak_speed":  108,
        "espeak_pitch":  25,
        "semitonos":     1.5,
        "chorus":        0.12,
        "chorus_detune": 0.10,
        "vibrato_hz":    3.5,
        "vibrato_amp":   0.006,
        "brillo":        0.12,
        "volumen":       0.90,
    },
    "bebe": {
        "descripcion":   "Mii bebé — agudísimo y adorable",
        "espeak_speed":  145,
        "espeak_pitch":  90,
        "semitonos":     14.0,
        "chorus":        0.32,
        "chorus_detune": 0.24,
        "vibrato_hz":    7.5,
        "vibrato_amp":   0.025,
        "brillo":        0.60,
        "volumen":       0.82,
    },
}

# ══════════════════════════════════════════════════════════════
#  MOTOR TTS
# ══════════════════════════════════════════════════════════════
class TomodachiTTS:

    def __init__(self, **cfg):
        self.cfg    = PRESETS["default"].copy()
        self.cfg.update(cfg)
        self.espeak = encontrar_espeak()
        self.ffmpeg = encontrar_ffmpeg()
        self.voz_es = encontrar_voz_es(self.espeak)

    # ── eSpeak: genera WAV base ───────────────────────────────
    def _espeak(self, texto: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            # FIX Windows: usar Popen con stdout/stderr como PIPE binario
            # y decodificar manualmente — evita el UnicodeDecodeError de cp1252
            proc = subprocess.Popen(
                [
                    self.espeak,
                    "-v", self.voz_es,
                    "-s", str(int(self.cfg["espeak_speed"])),
                    "-p", str(int(self.cfg["espeak_pitch"])),
                    "-a", "200",
                    "-w", tmp,
                    texto,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr_bytes = proc.communicate(timeout=30)
            if proc.returncode != 0:
                err = stderr_bytes.decode("utf-8", errors="ignore")
                raise RuntimeError(f"eSpeak error: {err}")

            sr, data = wavfile.read(tmp)

            if data.dtype == np.int16:
                audio = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                audio = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                audio = (data.astype(np.float32) - 128) / 128.0
            else:
                audio = data.astype(np.float32)

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Recortar silencio
            no_sil = np.where(np.abs(audio) > 0.005)[0]
            if len(no_sil):
                margen = int(sr * 0.05)
                audio  = audio[max(0, no_sil[0] - margen):min(len(audio), no_sil[-1] + margen)]

            return audio, int(sr)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    # ── DSP pipeline ─────────────────────────────────────────
    def _pitch_shift(self, audio, sr):
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=self.cfg["semitonos"])

    def _chorus(self, audio, sr):
        amt, detune = self.cfg["chorus"], self.cfg["chorus_detune"]
        if amt < 0.01:
            return audio
        l1 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=detune)
        l2 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-detune)
        return audio * (1 - amt) + l1 * (amt / 2) + l2 * (amt / 2)

    def _vibrato(self, audio, sr):
        hz, amp = self.cfg["vibrato_hz"], self.cfg["vibrato_amp"]
        if amp < 0.001 or hz < 0.1:
            return audio
        t = np.arange(len(audio)) / sr
        return audio * (0.94 + amp * np.sin(2 * np.pi * hz * t))

    def _eq(self, audio, sr):
        b = self.cfg["brillo"]
        if b < 0.01:
            return audio
        nyq = sr / 2
        sos = signal.butter(2, [1500 / nyq, 4000 / nyq], btype="band", output="sos")
        return audio + signal.sosfilt(sos, audio) * b

    def _formant_mii(self, audio, sr):
        """Realza el formante nasal característico del Mii (~900 Hz) y suaviza plosivos."""
        nyq = sr / 2
        # Boost nasal suave centrado en ~900 Hz
        sos_nasal = signal.butter(2, [700 / nyq, 1100 / nyq], btype="band", output="sos")
        audio = audio + signal.sosfilt(sos_nasal, audio) * 0.30
        # Corte de graves para quitar el boominess de eSpeak
        sos_hp = signal.butter(2, 180 / nyq, btype="high", output="sos")
        audio = signal.sosfilt(sos_hp, audio)
        # Suaviza plosivos con un low-pass muy ligero sobre transientes fuertes
        sos_lp = signal.butter(1, 7000 / nyq, btype="low", output="sos")
        suave  = signal.sosfilt(sos_lp, audio)
        mask   = (np.abs(audio) > 0.35).astype(np.float32)
        return audio * (1 - mask * 0.25) + suave * (mask * 0.25)

    def _lofi_mii(self, audio, sr):
        """Reduce ligeramente la calidad a ~11 kHz y vuelve a subir — da el carácter toy del Mii."""
        target_sr = 11025
        if sr <= target_sr:
            return audio
        step   = sr // target_sr
        crushed = audio[::step]
        # Repetir muestras para volver a la tasa original (hold interpolation)
        restored = np.repeat(crushed, step)[:len(audio)]
        # Mezcla sutil: 85% original + 15% lo-fi para conservar claridad
        return audio * 0.85 + restored * 0.15

    def _normalize(self, audio):
        audio = np.tanh(audio * 1.3) / 1.3
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx * self.cfg["volumen"]
        return audio.astype(np.float32)

    def sintetizar(self, texto: str):
        audio, sr = self._espeak(texto)
        audio = self._pitch_shift(audio, sr)
        audio = self._formant_mii(audio, sr)
        audio = self._lofi_mii(audio, sr)
        audio = self._chorus(audio, sr)
        audio = self._vibrato(audio, sr)
        audio = self._eq(audio, sr)
        audio = self._normalize(audio)
        return audio, sr

    # ── Reproducir ───────────────────────────────────────────
    def reproducir(self, texto: str):
        try:
            import sounddevice as sd
        except ImportError:
            print("❌ pip install sounddevice")
            return
        print(f'[reproduciendo] "{texto}"')
        audio, sr = self.sintetizar(texto)
        sd.play(audio, sr)
        sd.wait()

    # ── Guardar en cualquier formato ─────────────────────────
    def guardar(self, texto: str, ruta: str) -> str:
        """
        Guarda el audio en el formato indicado por la extensión del archivo.
        Formatos: wav, mp3, ogg, opus, m4a, flac
        Los formatos distintos a wav requieren ffmpeg instalado.
        """
        ext = os.path.splitext(ruta)[1].lower().lstrip(".")
        if not ext:
            ext = "wav"
            ruta += ".wav"

        if ext not in FORMATOS:
            print(f"⚠  Formato '.{ext}' no soportado. Usando WAV.")
            ext  = "wav"
            ruta = os.path.splitext(ruta)[0] + ".wav"

        # Siempre sintetizar a WAV primero
        audio, sr = self.sintetizar(texto)
        data16    = (audio * 32767).astype(np.int16)

        if ext == "wav":
            wavfile.write(ruta, sr, data16)
            print(f"[guardado] {ruta}")
            return ruta

        # Para otros formatos necesitamos ffmpeg o pydub
        if self.ffmpeg:
            return self._guardar_ffmpeg(data16, sr, ruta, ext)
        else:
            return self._guardar_pydub(data16, sr, ruta, ext)

    def _guardar_ffmpeg(self, data16: np.ndarray, sr: int, ruta: str, ext: str) -> str:
        """Convierte WAV → formato destino usando ffmpeg directamente."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            wavfile.write(tmp_wav, sr, data16)

            # Parámetros por formato
            extra = []
            if ext == "mp3":
                extra = ["-codec:a", "libmp3lame", "-q:a", "2"]
            elif ext == "ogg":
                extra = ["-codec:a", "libvorbis", "-q:a", "6"]
            elif ext == "opus":
                # WhatsApp usa opus en contenedor ogg a 48kHz
                extra = ["-codec:a", "libopus", "-b:a", "64k", "-ar", "48000"]
            elif ext == "m4a":
                extra = ["-codec:a", "aac", "-b:a", "128k"]
            elif ext == "flac":
                extra = ["-codec:a", "flac"]

            proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-i", tmp_wav] + extra + [ruta],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr_bytes = proc.communicate(timeout=60)
            if proc.returncode != 0:
                err = stderr_bytes.decode("utf-8", errors="ignore")
                raise RuntimeError(f"ffmpeg error:\n{err}")

            print(f"[guardado] {ruta}")
            return ruta
        finally:
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass

    def _guardar_pydub(self, data16: np.ndarray, sr: int, ruta: str, ext: str) -> str:
        """Fallback: usa pydub si ffmpeg no está en PATH."""
        try:
            from pydub import AudioSegment
        except ImportError:
            print("❌ Para exportar a {ext} instala pydub:  pip install pydub")
            print("   O instala ffmpeg: https://www.gyan.dev/ffmpeg/builds/")
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            wavfile.write(tmp_wav, sr, data16)
            seg = AudioSegment.from_wav(tmp_wav)

            fmt_map = {"mp3": "mp3", "ogg": "ogg", "opus": "opus",
                       "m4a": "mp4", "flac": "flac"}
            fmt = fmt_map.get(ext, ext)

            params = []
            if ext == "opus":
                seg = seg.set_frame_rate(48000)  # WhatsApp requiere 48kHz

            seg.export(ruta, format=fmt, parameters=params)
            print(f"[guardado] {ruta}")
            return ruta
        finally:
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def build_parser():
    formatos_str = ", ".join(FORMATOS.keys())
    p = argparse.ArgumentParser(
        description="🎮 Tomodachi TTS — voz estilo Tomodachi Life",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Formatos de salida disponibles: {formatos_str}

Ejemplos:
  python tomodachi_tts.py "Hola! Me llamo Mii!"
  python tomodachi_tts.py "Texto" --preset kawaii
  python tomodachi_tts.py "Texto" --guardar voz.wav
  python tomodachi_tts.py "Texto" --guardar voz.mp3
  python tomodachi_tts.py "Texto" --guardar voz.opus       <- WhatsApp
  python tomodachi_tts.py "Texto" --semitonos 9 --chorus 0.3
  python tomodachi_tts.py --listar-presets
  python tomodachi_tts.py --listar-formatos
        """,
    )
    p.add_argument("texto", nargs="?", help="Texto a sintetizar")
    p.add_argument("--preset", default="default", choices=list(PRESETS.keys()),
                   help="Preset de voz (por defecto: %(default)s)")

    g = p.add_argument_group("Ajuste de parámetros")
    g.add_argument("--semitonos",  type=float, metavar="N", help="Pitch shift en semitonos (default: 7)")
    g.add_argument("--velocidad",  type=int,   metavar="N", help="Palabras por minuto (80–250)")
    g.add_argument("--tono",       type=int,   metavar="N", help="Tono base eSpeak (0–99)")
    g.add_argument("--chorus",     type=float, metavar="N", help="Grosor del chorus 0.0–1.0")
    g.add_argument("--vibrato",    type=float, metavar="N", help="Intensidad del vibrato 0.0–0.1")
    g.add_argument("--vibrato-hz", type=float, metavar="N", help="Velocidad del vibrato en Hz")
    g.add_argument("--brillo",     type=float, metavar="N", help="Realce de agudos 0.0–1.0")
    g.add_argument("--volumen",    type=float, metavar="N", help="Volumen final 0.0–1.0")

    p.add_argument("--guardar",        metavar="archivo.ext",
                   help=f"Guardar audio ({formatos_str})")
    p.add_argument("--listar-presets", action="store_true", help="Muestra los presets y sale")
    p.add_argument("--listar-formatos",action="store_true", help="Muestra los formatos soportados y sale")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.listar_formatos:
        print(f"\n  {'EXT':<6}  {'DESCRIPCION':<45}  {'FFMPEG?':>8}")
        print("  " + "─" * 62)
        for ext, info in FORMATOS.items():
            req = "requerido" if info["requiere_ffmpeg"] else "no"
            print(f"  {ext:<6}  {info['desc']:<45}  {req:>8}")
        print()
        return

    if args.listar_presets:
        print(f"\n  {'PRESET':<10}  {'DESCRIPCION':<45}  {'SEMIT':>5}  {'VEL':>4}")
        print("  " + "─" * 70)
        for name, cfg in PRESETS.items():
            print(f"  {name:<10}  {cfg['descripcion']:<45}  "
                  f"{cfg['semitonos']:>4.0f}st  {cfg['espeak_speed']:>3}wpm")
        print()
        return

    cfg = PRESETS[args.preset].copy()
    overrides = {
        "semitonos":  "semitonos",
        "velocidad":  "espeak_speed",
        "tono":       "espeak_pitch",
        "chorus":     "chorus",
        "vibrato":    "vibrato_amp",
        "vibrato_hz": "vibrato_hz",
        "brillo":     "brillo",
        "volumen":    "volumen",
    }
    for arg_key, cfg_key in overrides.items():
        val = getattr(args, arg_key.replace("-", "_"), None)
        if val is not None:
            cfg[cfg_key] = val

    try:
        tts = TomodachiTTS(**cfg)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not args.texto:
        parser.print_help()
        sys.exit(0)

    ext = os.path.splitext(args.guardar)[1].lstrip(".").lower() if args.guardar else "—"
    print(f"\nLobotoMII  |  preset: {args.preset}  |  "
          f"semitonos: {cfg['semitonos']}  |  vel: {cfg['espeak_speed']}wpm  |  "
          f"formato: {ext}\n")

    if args.guardar:
        tts.guardar(args.texto, args.guardar)
    else:
        tts.reproducir(args.texto)


if __name__ == "__main__":
    main()