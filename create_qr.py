"""Create a PNG QR for the *real* public Streamlit URL after deployment.
Usage: py create_qr.py https://your-real-app.streamlit.app
Install QR package once: py -m pip install "qrcode[pil]"
"""
import sys
from urllib.parse import urlparse
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: py create_qr.py https://your-app.streamlit.app")
url = sys.argv[1].strip()
parsed = urlparse(url)
if parsed.scheme != "https" or not parsed.netloc:
    raise SystemExit("Please enter a real https:// URL of your deployed app (not localhost:8501).")
try:
    import qrcode
except ImportError as exc:
    raise SystemExit('Please run: py -m pip install "qrcode[pil]"') from exc
image = qrcode.make(url)
output = Path(__file__).resolve().parent / "HealthCover_QR.png"
image.save(output)
print("QR code saved:", output)
print("Destination:", url)
