"""Extract the cited coordination steps from the submitted rollout GIF."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
STEPS = (5, 6, 8, 13)

def main():
    with Image.open(HERE.parents[1] / "final_policy.gif") as gif:
        width, height = gif.size
        header = 44
        strip = Image.new("RGB",(2*width,2*(height + header)),"white")
        draw = ImageDraw.Draw(strip)
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"),28)
        for index, step in enumerate(STEPS):
            gif.seek(step)
            frame = ImageOps.expand(gif.convert("RGB"),border=2,fill="#cccccc").resize((width,height))
            x, y = (index % 2)*width, (index // 2)*(height + header)
            draw.text((x + 18,y + 6),f"Step {step}",font=font,fill="black")
            strip.paste(frame,(x,y + header))
        strip.save(HERE / "coordination_frames.png",dpi=(300,300))

if __name__ == "__main__":
    main()
