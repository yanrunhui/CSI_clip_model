from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("/Users/yanrunhui/CSI_model/output/pdf/seen_vs_unseen_nf128_combined.png")
W, H = 4800, 1400
BLUE = "#3C5488"
RED = "#E64B35"
INK = "#2B2B2B"
GRID = "#E6E6E6"
WHITE = "#FFFFFF"
FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


img = Image.new("RGB", (W, H), WHITE)
draw = ImageDraw.Draw(img)


def centered(text, xy, fnt, fill=INK):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)


def rotated_centered(text, center, fnt):
    box = fnt.getbbox(text)
    tw, th = box[2] - box[0], box[3] - box[1]
    layer = Image.new("RGBA", (tw + 30, th + 30), (255, 255, 255, 0))
    ld = ImageDraw.Draw(layer)
    ld.text((15, 10 - box[1]), text, font=fnt, fill=INK)
    layer = layer.rotate(90, expand=True)
    img.paste(layer, (int(center[0] - layer.width / 2), int(center[1] - layer.height / 2)), layer)


def grouped_panel(box, labels, seen, unseen, seen_err, unseen_err, ylabel, title, yrange, ticks):
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 230, x1 - 35, y0 + 85, y1 - 165
    draw.text((x0 + 10, y0), title, font=font(90, True), fill=INK)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    ymin, ymax = yrange

    def py(v):
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)

    for tick in ticks:
        yy = py(tick)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        label = f"{tick:g}"
        tb = draw.textbbox((0, 0), label, font=font(74))
        draw.text((left - 18 - (tb[2] - tb[0]), yy - 41), label, font=font(74), fill=INK)

    rotated_centered(ylabel, (x0 + 58, (top + bottom) / 2), font(76))
    n = len(labels)
    group_w = (right - left) / n
    bar_w = min(105, group_w * 0.29)
    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        for offset, value, error, color in (
            (-bar_w * 0.58, seen[i], seen_err[i], BLUE),
            (bar_w * 0.58, unseen[i], unseen_err[i], RED),
        ):
            bx0, bx1 = cx + offset - bar_w / 2, cx + offset + bar_w / 2
            by = py(value)
            draw.rectangle((bx0, by, bx1, bottom), fill=color, outline=INK, width=2)
            if error is not None:
                ey0, ey1 = py(value + error), py(max(ymin, value - error))
                ex = (bx0 + bx1) / 2
                draw.line((ex, ey0, ex, ey1), fill=INK, width=3)
                draw.line((ex - 14, ey0, ex + 14, ey0), fill=INK, width=3)
                draw.line((ex - 14, ey1, ex + 14, ey1), fill=INK, width=3)
                label_y = ey0 - 52
            else:
                label_y = by - 52
            value_text = f"{value:.2f}"
            vb = draw.textbbox((0, 0), value_text, font=font(43, True))
            draw.text(((bx0 + bx1) / 2 - (vb[2] - vb[0]) / 2, label_y - 8), value_text, font=font(43, True), fill=INK)
        lines = label.split("\n")
        for j, line in enumerate(lines):
            centered(line, (cx, bottom + 18 + j * 70), font(68))


centered("Generalization from Seen to Unseen Nf = 128", (W / 2, 16), font(86, True))
legend_y = 112
legend_font = font(86)
legend_items = [
    (BLUE, "Main model (seen Nf = 128)"),
    (RED, "Held-out model (unseen Nf = 128)"),
]
starts = [1420, 2720]
for start, (color, text) in zip(starts, legend_items):
    draw.rectangle((start, legend_y + 10, start + 55, legend_y + 65), fill=color, outline=INK, width=2)
    draw.text((start + 74, legend_y), text, font=legend_font, fill=INK)

grouped_panel(
    (70, 225, 1450, 1325),
    ["LoS/NLoS\naccuracy", "Text\nfactuality", "Numeric\naccuracy"],
    [99.58, 96.43, 82.61],
    [99.827, 95.667, 78.639],
    [0.11, None, None],
    [0.006, 0.179, 0.906],
    "Score (%)",
    "(a) Accuracy and text quality",
    (72, 104),
    [75, 80, 85, 90, 95, 100],
)

grouped_panel(
    (1510, 225, 3575, 1325),
    ["First-angle\n(°)", "First-delay\n(ns)", "LoS delay\n(ns)", "NLoS delay\n(ns)"],
    [16.61, 21.91, 8.83, 40.66],
    [20.483, 58.752, 21.801, 92.39],
    [0.56, 0.30, 0.55, 1.58],
    [0.146, 1.338, 1.270, None],
    "MAE (native unit)",
    "(b) Delay and angle errors",
    (0, 105),
    [0, 20, 40, 60, 80, 100],
)

grouped_panel(
    (3635, 225, 4760, 1325),
    ["First-path\npower", "K-factor"],
    [4.34, 2.15],
    [5.998, 2.166],
    [0.27, 0.04],
    [0.355, 0.037],
    "MAE (dB)",
    "(c) Power and K-factor errors",
    (0, 7),
    [0, 1, 2, 3, 4, 5, 6, 7],
)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=96, dpi=(300, 300))
print(OUT)
