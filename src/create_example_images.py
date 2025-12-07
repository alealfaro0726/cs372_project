
from PIL import Image, ImageDraw
import numpy as np


def create_warm_sunset():
    img = Image.new('RGB', (512, 512))
    draw = ImageDraw.Draw(img)

    for y in range(512):
        ratio = y / 512
        r = int(255 * (1 - ratio * 0.3))
        g = int(150 * (1 - ratio * 0.5))
        b = int(50 * (1 - ratio * 0.7))
        draw.rectangle([(0, y), (512, y+1)], fill=(r, g, b))

    return img


def create_cool_ocean():
    img = Image.new('RGB', (512, 512))
    draw = ImageDraw.Draw(img)

    for y in range(512):
        ratio = y / 512
        r = int(30 + 50 * ratio)
        g = int(100 + 100 * ratio)
        b = int(180 + 50 * (1 - ratio))
        draw.rectangle([(0, y), (512, y+1)], fill=(r, g, b))

    return img


def create_abstract_energetic():
    img = Image.new('RGB', (512, 512))
    pixels = np.array(img)

    for i in range(0, 512, 50):
        for j in range(0, 512, 50):
            color = (
                np.random.randint(150, 255),
                np.random.randint(50, 200),
                np.random.randint(0, 150)
            )
            pixels[i:i+50, j:j+50] = color

    return Image.fromarray(pixels.astype('uint8'))


if __name__ == '__main__':
    import os
    os.makedirs('../data/examples', exist_ok=True)

    create_warm_sunset().save('../data/examples/warm_sunset.jpg')
    create_cool_ocean().save('../data/examples/cool_ocean.jpg')
    create_abstract_energetic().save('../data/examples/abstract_energetic.jpg')

    print("Created 3 example images in data/examples/ directory")
