from PIL import Image, ImageDraw, ImageFilter
import face_recognition
from shapely import Point, LineString, Polygon
import geopandas
import matplotlib.path as mplPath
import numpy as np
import sys
import cv2
import random

# rotate
def rotate(img_path, deg):
    return Image.open(img_path).rotate(deg) 

# blur
def blur(img_path, radius):
    return Image.open(img_path).filter(ImageFilter.GaussianBlur(radius))

# increase contrast

# lighten

# darken
def adjust(img_path, contrast=1, brightness=0):
    img = cv2.imread(img_path)
    arr = cv2.addWeighted(img, contrast, img, 0, brightness)
    # print(arr)
    # return arr
    return Image.fromarray(np.array([[[r, g, b] for [b, g, r] in line] for line in arr]))

def adjustBW(img_path, contrast=1, brightness=0):
    img = cv2.imread(img_path)
    arr = cv2.addWeighted(img, contrast, img, 0, brightness)
    # print(arr)
    # return arr
    return Image.fromarray(arr)

def darken_area(input_img, radius, poly_arr):
    pixel_map = input_img.load()

    w, h = input_img.size

    poly_srs_arr = [geopandas.GeoSeries(polygon) for polygon in poly_arr]

    for i in range(w):
        for j in range(h):
            r, g, b = input_img.getpixel((i, j))
            ratio = 1
            for k in range(len(poly_arr)):
                poly = poly_arr[k]
                poly_srs = poly_srs_arr[k]
                if poly.contains(Point(i, j)):
                    ratio = 0.75
                else:
                    dist = poly_srs.distance(geopandas.GeoSeries(Point(i, j)))[0]
                    if dist < radius:
                        ratio = 0.5 + dist/(2*radius)
            pixel_map[i, j] = (int(ratio*r), int(ratio*g), int(ratio*b))

    return input_img

# darken area around eyes
# https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py
# https://www.geeksforgeeks.org/how-to-manipulate-the-pixel-values-of-an-image-using-python/ 
# https://www.tutorialspoint.com/what-s-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# https://www.geeksforgeeks.org/python-sympy-polygon-distance-method/
def darken_eyes(img_path, radius):
    img = face_recognition.load_image_file(img_path)
    landmarks = face_recognition.face_landmarks(img)[0]
    poly_arr = [Polygon(landmarks['left_eye']),  Polygon(landmarks['right_eye'])]
    return darken_area(input_img=Image.open(img_path), radius=radius, poly_arr=poly_arr)

def darken_shadows(img_path, radius):
    img = face_recognition.load_image_file(img_path)
    landmarks = face_recognition.face_landmarks(img)[0]
    landmark_names = ['left_eyebrow', 'right_eyebrow', 'nose_tip', 'chin']
    poly_arr = [LineString(landmarks[landmark_name]) for landmark_name in landmark_names]
    return darken_area(input_img=Image.open(img_path), radius=radius, poly_arr=poly_arr)

def clown_makeup(img_path, opacity, radius):
    input_img=Image.open(img_path)
    img = face_recognition.load_image_file(img_path)
    pixel_map = input_img.load()
    w, h = input_img.size

    landmarks = face_recognition.face_landmarks(img)[0]
    # print(landmarks.keys())

    poly_arr = [Polygon(landmarks['left_eye']),  
                Polygon(landmarks['right_eye']),  
                Polygon(landmarks['top_lip']),  
                Polygon(landmarks['bottom_lip'])]
    
    poly_srs_arr = [geopandas.GeoSeries(polygon) for polygon in poly_arr]
    # print(poly_srs_arr)

    for i in range(w):
        for j in range(h):
            r, g, b = input_img.getpixel((i, j))
            within = False
            min_dist = max(w, h)
            for k in range(len(poly_arr)):
                poly = poly_arr[k]
                poly_srs = poly_srs_arr[k]
                if poly.contains(Point(i, j)):
                    within = True
                curr_dist = poly_srs.distance(geopandas.GeoSeries(Point(i, j)))[0]
                min_dist = min(min_dist, curr_dist)

            old_colors  = [(1 - opacity)*color for color in (r, g, b)]

            if within:
                new_colors = [opacity*255, 0, 0]

            elif min_dist < radius:
                quarter = radius/4
                if min_dist < 2*quarter:
                    ratio = (min_dist)/(2*quarter)
                    new_r = (opacity) * (1 - ratio) * 255
                    new_b = (opacity) * ratio * 255
                    new_g = 0
                elif min_dist < 3*quarter:
                    ratio = (min_dist - (2*quarter))/(quarter)
                    new_r = 0
                    new_b = (opacity) * (1 - ratio) * 255
                    new_g = (opacity) * ratio * 255
                else:
                    ratio = (min_dist - 3*quarter)/(quarter)
                    new_r = (opacity) * ratio * r
                    new_b = (opacity) * ratio * b
                    new_g = ((opacity) * (1 - ratio) * 255) + ((opacity) * (ratio) * g)
                new_colors = (new_r, new_g, new_b)
            else:
                new_colors = [opacity*color for color in (r, g, b)]
            
            res_colors = [int(old_colors[i] + new_colors[i]) for i in range(3)]
            pixel_map[i, j] = (res_colors[0], res_colors[1], res_colors[2])

    return input_img

def add_shadow(drawing, line, opacity, up):
    add = 1 if up else -1
    drawing.line(line, fill=(0, 0, 0, opacity), width=1)
    return [(x, y + add) for (x, y) in line]

def darken_shadows_fast(img_path, opacity, radius):
    image = face_recognition.load_image_file(img_path)
    landmarks = face_recognition.face_landmarks(image)[0]
    pil_image = Image.fromarray(image)
    drawing = ImageDraw.Draw(pil_image, 'RGBA')


    for i in range(radius):
        curr_opacity = int(255*opacity*(1 - i/radius))
        landmarks['left_eyebrow'] = add_shadow(drawing=drawing, 
                                               line=landmarks['left_eyebrow'], 
                                               opacity=curr_opacity, 
                                               up=True)
        landmarks['right_eyebrow'] = add_shadow(drawing=drawing, 
                                                line=landmarks['right_eyebrow'], 
                                                opacity=curr_opacity, 
                                                up=True)
        landmarks['nose_tip'] = add_shadow(drawing=drawing, 
                                           line=landmarks['nose_tip'], 
                                           opacity=curr_opacity, 
                                           up=True)
        landmarks['chin'] = add_shadow(drawing=drawing, 
                                       line=landmarks['chin'], 
                                       opacity=curr_opacity, 
                                       up=False)

    return pil_image

def add_random(img_path):
    w, h = Image.open(img_path).size
    img = cv2.imread(img_path)

    new_arr = []

    for i in range(h):
        row = []
        for j in range(w):
            row.append((np.uint8(random.randint(0,255)),
                        np.uint8(random.randint(0,255)), 
                        np.uint8(random.randint(0,255))))
        new_arr.append(row)
            
    new_arr = np.array(new_arr)
    res = cv2.add(img, new_arr)
    return Image.fromarray(np.array([[[r, g, b] for [b, g, r] in line] for line in res]))

def add_random2(img_path):
    input_img=Image.open(img_path)
    pixel_map = input_img.load()
    w, h = input_img.size

    for i in range(w):
        for j in range(h):
            r, g, b = input_img.getpixel((i, j))
            opacity = random.randint(0, 100)/100
            old_colors = [opacity*val for val in [r, g, b]]
            new_colors = [(1 - opacity)*(random.randint(0, 255)) for _ in range(3)]
            res_colors = [int(old_colors[i] + new_colors[i]) for i in range(3)]
            pixel_map[i, j] = (res_colors[0], res_colors[1], res_colors[2])

    return input_img


# rotate("chloe2.jpg", 45).save("chloe2rot.jpg")
# blur("chloe2.jpg", 50).save("chloe2blur.jpg")
# adjust("chloe2.jpg", contrast=2).save("chloe2contrast.jpg")
# adjust("chloe2.jpg", brightness=-50).save("chloe2dark.jpg")
# adjust("chloe2.jpg", brightness=50).save("chloe2light.jpg")
# darken_eyes("chloe2.jpg", 15).save("chloe2darkeyes1.jpg")
# darken_shadows("chloe2.jpg", 15).save("chloe2darkchin.jpg")
# clown_makeup("chloe2.jpg", opacity = 0.2, radius = 20).save("chloe2clown.jpg")

# darken_shadows_fast("chloe1.jpg", opacity=0.5, radius=20).save("chloe1darkbrow.jpg")
# darken_shadows_fast("chloe2.jpg", opacity=0.5, radius=20).save("chloe2darkbrow.jpg")
# darken_shadows_fast("chloe3.jpg", opacity=0.5, radius=20).save("chloe3darkbrow.jpg")

# add_random("chloe2.jpg").save("chloe2noisy.jpg")
# add_random2("chloe2.jpg").save("chloe2noisy2.jpg")