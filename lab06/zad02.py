from PIL import Image

# file = "C:\\Users\\marta\\Pictures\\mewa.jpg"
# file = "C:\\Users\\marta\\Pictures\\wallpaper\\3d3f7995a8d742ec23b50ca1d9375d3e.jpg"
# file = "C:\\Users\\marta\\Pictures\\wallpaper\\1316319.jpeg"
# file = "C:\\Users\\marta\\Pictures\\wallpaper\\pexels-eberhard-grossgasteiger-691668.jpg"
file = "C:\\Users\\marta\\Pictures\\wallpaper\\nika-benedictova-_913XLNbMJ0-unsplash.jpg"

img = Image.open(file)
img_data = img.getdata()

lst = []
for i in img_data:
    lst.append(round((i[0] + i[1] + i[2]) / 3))

new_img = Image.new("L", img.size)
new_img.putdata(lst)

new_img.show()

lst = []
for i in img_data:
    lst.append(i[0] * 0.299 + i[1] * 0.587 + i[2] * 0.114)

new_img = Image.new("L", img.size)
new_img.putdata(lst)

new_img.show()
