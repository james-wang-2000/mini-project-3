from PIL import Image

def convert(image):
    data = list()
    bmp_image = Image.open(image)

    for j in range(8):
        for i in range(8):
            gray = (255 - bmp_image.getpixel((i,j))[0])
            data.append(gray)

    return data

if __name__ == '__main__':
    print(convert(r"C:\Users\Admin\Desktop\Python\計程實驗\mini_project_3\2.png"))
    
    
