""" This script is used to preprocess an image bank to make all the 
images the same size. Images are first scaled and then buffered (if
they are not square) with a color approximating their background. 
Edit line 34 to change the output size. 
Edit line 38 to change the input directory containing images to be processed. 
Edit line 39 to change the output directory. 

"""
import PIL
from PIL import Image,ImageOps
import os
from numpy import mean

def isEven(n):
    return n % 2 == 0

def getBorderColor(img):
    image_size = img.size
    R = []
    G = []
    B = []
    for i in range(image_size[0]):
        r, g, b = img.getpixel((i, 0))
        R.append(r); G.append(g); B.append(b)
        r, g, b = img.getpixel((i, image_size[1]-1))
        R.append(r); G.append(g); B.append(b)
    
    for i in range(image_size[1]):
        r, g, b = img.getpixel((0, i))
        R.append(r); G.append(g); B.append(b)
        r, g, b = img.getpixel((image_size[0]-1, i))
        R.append(r); G.append(g); B.append(b)
    
    return (int(mean(R)), int(mean(G)), int(mean(B)))

output_size = (128,128)
#image_dir = r'D:\IMEA\Tasking\2017\LearningNeuralNets\MyExample\images\merchendise'
#output_dir = os.path.join(image_dir,'merchendise_preprocessed')
image_dir = r'D:\IMEA\Tasking\2017\LearningNeuralNets\MyExample\images\shoes'
output_dir = os.path.join(image_dir,'shoes_preprocessed')
os.mkdir(output_dir)

image_names = os.listdir(image_dir)

picture_index = 0
for image_name in image_names:
    try:
        image_filename = os.path.join(image_dir,image_name)
        img = Image.open(image_filename)
        
        scale = output_size[0] / max(img.size)
        new_size = [int(scale*img.size[0]), int(scale*img.size[1])]
        if not isEven(new_size[0]): new_size[0] -= 1
        
        if not isEven(new_size[1]): new_size[1] -= 1
        
        img = img.resize(new_size, PIL.Image.ANTIALIAS)
        
        border = ( int((output_size[0]-new_size[0])/2.), int((output_size[1]-new_size[1])/2.) )
        fill = getBorderColor(img)
        img_with_border = ImageOps.expand(img, border=border, fill=fill)
        output_file = os.path.join(output_dir,'%04d.jpg' % picture_index)
        img_with_border.save(output_file)
        picture_index += 1
    except:
        continue


