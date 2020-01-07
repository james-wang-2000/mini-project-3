from PIL import Image

def convert(image):
    data = list()
    bmp_image = Image.open(image)

    for j in range(8):
        for i in range(8):
            gray = (255 - bmp_image.getpixel((i,j))[0])
            data.append(gray)

    return data

