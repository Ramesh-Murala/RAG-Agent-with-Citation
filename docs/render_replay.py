"""Render an animated transcript from captured request/response JSON (requires Pillow)."""
from pathlib import Path
import json, textwrap
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).parent
config = json.loads((ROOT / "assets" / "capture.json").read_text())
request = (ROOT / "assets" / "request.json").read_text().strip()
response = (ROOT / "assets" / "response.json").read_text().strip()
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
font = ImageFont.truetype(font_path, 20)
small = ImageFont.truetype(font_path, 17)
def lines(text):
    return [part for line in text.splitlines() for part in (textwrap.wrap(line, 85, subsequent_indent="  ", replace_whitespace=False) or [""])]
frames_text = [config["request_label"] + "\n\n" + request, config["response_label"] + "\n\n" + response]
height = max(520, max(len(lines(t)) for t in frames_text) * 28 + 155)
frames=[]
for i, text in enumerate(frames_text):
    im=Image.new("RGB", (1120,height), "#0b1422")
    d=ImageDraw.Draw(im)
    d.rectangle((0,0,1120,70),fill="#14243a")
    d.text((32,24),config["title"],font=font,fill="#e8f0fa")
    d.text((32,88),config["mode"],font=small,fill="#82b9cf")
    for n,line in enumerate(lines(text)):
        d.text((32,137+n*28),line,font=font,fill="#9de8bf" if n==0 else "#e0e7f0")
    frames.append(im)
frames[0].save(ROOT / "assets" / "api-demo.gif",save_all=True,append_images=frames[1:],duration=[3000,6500],loop=0)
frames[1].save(ROOT / "assets" / "response-preview.png")
