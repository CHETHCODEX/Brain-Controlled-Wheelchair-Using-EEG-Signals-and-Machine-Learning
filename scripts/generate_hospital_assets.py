"""Render the clinic floor material used by Webots and its matching dashboard map."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def generate():
    image = Image.new("RGB", (1800, 1200), "#edf1f0")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/segoeuib.ttf")
    font = ImageFont.truetype(str(font_path), 26) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for column in range(0, 1800, 50):
        draw.line((column, 0, column, 1200), fill="#dde5e2", width=1)
    for row in range(0, 1200, 50):
        draw.line((0, row, 1800, row), fill="#dde5e2", width=1)
    draw.rectangle((0, 420, 1800, 780), fill="#f8faf9")
    rooms = [(0, 0, 500, 420, "01  WARD A", "#dcece9"),
             (500, 0, 900, 420, "02  WARD B", "#e1eaf0"),
             (900, 0, 1300, 420, "03  ASSESSMENT", "#e6e9df"),
             (1300, 0, 1800, 420, "04  REHAB GYM", "#dbebe3"),
             (0, 780, 600, 1200, "05  CONSULTATION", "#e1eaf0"),
             (600, 780, 1200, 1200, "06  QUIET ROOM", "#e7e9e1"),
             (1200, 780, 1800, 1200, "07  CARE STATION", "#dcece9")]
    for left, top, right, bottom, name, color in rooms:
        draw.rectangle((left + 12, top + 12, right - 12, bottom - 12), fill=color)
        draw.text((left + 35, top + 35), name, font=font, fill="#345b59")
    for start in range(80, 1740, 70):
        draw.line((start, 605, start + 38, 605), fill="#4e9991", width=6)
    for destination in (300, 700, 1100, 1500):
        draw.line((destination, 605, destination, 420), fill="#94bdb4", width=3)
        draw.ellipse((destination - 15, 435, destination + 15, 465), fill="#157d72", outline="white", width=4)
    draw.text((730, 680), "REHABILITATION CORRIDOR", font=small, fill="#66837c")
    draw.ellipse((150, 550, 250, 650), outline="#157d72", width=5)
    draw.text((125, 660), "START / DOCK", font=small, fill="#345b59")
    output = ROOT / "simulation" / "textures" / "hospital_floor.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((2048, 2048), Image.Resampling.LANCZOS).save(output)
    print(output)


if __name__ == "__main__":
    generate()