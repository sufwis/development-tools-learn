# to test Pillow
from PIL import Image,ImageEnhance
from pathlib import Path
import time

image_path = Path("1.png")
if not image_path.exists():
    image_path  = Path("py&cmd-line-env\\figure\\alias.png")
images = []

# part 1
print("Image object")
alias_image = Image.open(str(image_path))
new_image = Image.new('RGB',(400,300),'red')
alias_image.save("py&cmd-line-env\\py-cod-test\\1.png",optimize=True)
new_image.save("py&cmd-line-env\\py-cod-test\\2.png",optimize=True)

images.append(alias_image)
images.append(new_image)

def show_attr(img):
    print(img.format) 
    print(img.size) 
    print(img.width)
    print(img.height) 
    print(img.mode)
    # img.show()
    # time.sleep(3)

print("image1 attr:")
show_attr(images[0])

print("image2 attr:")
show_attr(images[1])

# part2
# 1转换mode
images.pop(1)
gray_img = images[0].convert('L')
rgba_img = images[0].convert('RGBA')
# gray_img.show()
# time.sleep(3)
# rgba_img.show()
# time.sleep(3)

# 2缩放
resized_img = images[0].resize((900,600),Image.LANCZOS)
# resized_img.show()
# time.sleep(3)
copy_img = images[0].copy()
copy_img.thumbnail((150,150))
# copy_img.show()
# time.sleep(3)

# 3剪切
# images[0].crop((100,300,200,400)).show()

# 4旋转和翻转
# images[0].rotate(90,expand=True,fillcolor='white').show()
# images[0].transpose(Image.FLIP_TOP_BOTTOM).show()
# images[0].transpose(Image.FLIP_LEFT_RIGHT).show()

# 5粘贴
image2 = Image.open("py&cmd-line-env\\py-cod-test\\2.png")
copy_img = images[0].copy()
copy_img.paste(image2,(50,50))
# copy_img.show()


# part3 图像增强
# 1对比度

def enhance_and_show(img):
    ImageEnhance.Contrast(img).enhance(2).show()
    time.sleep(2.5)
    ImageEnhance.Brightness(img).enhance(2).show()
    time.sleep(2.5)
    ImageEnhance.Color(img).enhance(2).show()
    time.sleep(2.5)
    ImageEnhance.Sharpness(img).enhance(2).show()
    time.sleep(2.5)    

enhance_and_show(images[0])