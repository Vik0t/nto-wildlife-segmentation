from PIL import Image

def is_grey_scale(img_path):
    counter = 0
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                counter += 1
    if counter > 100000:
        return False
    return True

