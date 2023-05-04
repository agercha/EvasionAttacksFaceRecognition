from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

import face_recognition
import os.path
from transforms import *
import random

import uuid
import copy
import math



num_filters = 50 
training_iters = 500 

weights = [i for i in range(num_filters)] 

class Face():
    def __init__(self, sub_num, lighting, view, gender, glasses, transform, encoding):
        self.sub_num = sub_num
        self.lighting = lighting
        self.view = view
        self.gender = gender
        self.glasses = glasses
        self.encoding = encoding
        self.transform = transform.name

    def __repr__(self):
        return f"{self.sub_num}\t{self.lighting}\t{self.view}\t{self.gender}\t{self.glasses}\t{self.transform}\t"
    
    def get_str(self):
        return f"{self.sub_num}\t{self.lighting}\t{self.view}\t{self.gender}\t{self.glasses}\t{self.transform}\n"

class Dist():
    def __init__(self, dist, recognized, face1, face2):
        self.dist = dist
        self.recognized = recognized
        self.face1 = face1
        self.face2 = face2

    def __repr__(self):
        return f"{self.dist}\t{self.recognized}\t{self.face1}{self.face2}\n"
    
    def get_str(self):
        return f"{self.dist}\t{self.recognized}\t{self.face1}{self.face2}\n"
    

def doNothing(img_path):
    return Image.open(img_path)

class Transform():
    def __init__(self, name, func):
        self.name = name
        self.func = func

class FilterIm():
    def __init__(self, rectangles, method, iter):
        self.rectangles = rectangles
        self.method = method
        self.dist = None
        self.iter = iter

    def __repr__(self):
        return f'{self.dist}'
    
class Rectangle():
    def __init__(self, start, end, color, thick):
        self.start = start
        self.end = end
        self.color = color
        self.thickness = thick

curr_path = f"../muct/i000qa-fn.jpg"
img = face_recognition.load_image_file(curr_path)
enc_arr = face_recognition.face_encodings(img)
face = Face(0, "q", "a", "f", "n", Transform("none", doNothing), enc_arr[0])
top_filters_overall = []
input_img=Image.open(curr_path)
w, h = Image.open(curr_path).size
max_cover = int(w*h/100)


def create_filter(iter):
    filled = 0
    rects = []
    max_w = int(w/10)
    max_h = int(h/10)
    max_thick = int(w/25)
    while filled < max_cover:
        x0 = random.randint(0, w)
        y0 = random.randint(0, h)
        width = random.randint(0, max_w)
        height = random.randint(0, max_h)
        x1 = min(x0 + width, w)
        y1 = min(y0 + height, h)
        color = (random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255))
        if random.randint(0,1):
            thickness = -1
            filled += abs((x1-x0)*(y1-y0))
        else:
            thickness = random.randint(0,max_thick)
            filled += 2*thickness*(abs(x1 - x0) + abs(y1 - y0) + 2*thickness)
        curr_rect = Rectangle((x0, y0), (x1, y1), color, thickness)
        rects.append(curr_rect)
    if len(rects) > 0: rects.pop(-1)
    return FilterIm(rects, "random", iter)


def perturb_filter(f, iter):
    filled = 0
    rects = []
    w_pertub = int(w/20)
    h_pertub = int(h/20)
    while filled < max_cover:
        rect = random.randchoice(f.rectangles)
        x0 = min(w, max(rect.start[0] + random.randint(-w_pertub, w_pertub), 0))
        y0 = min(h, max(rect.start[1] + random.randint(-h_pertub, h_pertub), 0))
        x1 = min(w, max(rect.end[0] + random.randint(-w_pertub, w_pertub), 0))
        y1 = min(h, max(rect.end[1] + random.randint(-h_pertub, h_pertub), 0))
        r = max(0, min(255, rect.color[0] + random.randint(-50, 50)))
        g = max(0, min(255, rect.color[1] + random.randint(-50, 50)))
        b = max(0, min(255, rect.color[2] + random.randint(-50, 50)))
        color = (r, g, b)
        thickness = rect.thickness
        if thickness == -1:
            filled += abs((x1-x0)*(y1-y0))
        else:
            filled += 2*thickness*(abs(x1 - x0) + abs(y1 - y0) + 2*thickness)
        curr_rect = Rectangle((x0, y0), (x1, y1), color, thickness)
        rects.append(curr_rect)
    if len(rects) > 0: rects.pop(-1)
    return FilterIm(rects, "perturb", iter)

def mate_filters(f0, f1, iter):
    filled = 0
    rects_old = copy.deepcopy(f0.rectangles) + copy.deepcopy(f1.rectangles)
    rects = []
    i = 0
    while filled < max_cover and i < 20:
        i += 1
        curr_rect = random.choice(rects_old)
        (x0, y0) = curr_rect.start
        (x1, y1) = curr_rect.end
        thickness = curr_rect.thickness
        if thickness == -1:
            filled += abs((x1-x0)*(y1-y0))
        else:
            filled += 2*thickness*(abs(x1 - x0) + abs(y1 - y0) + 2*thickness)
        rects.append(curr_rect)
    if len(rects) > 0: rects.pop(-1)
    return FilterIm(rects, "mate", iter)

def apply_filter(f, res_path="tmp_randomfuzz3.jpg", with_arrow=False):
    image = cv2.imread(curr_path)
    for rect in f.rectangles:
        cv2.rectangle(image, rect.start, rect.end, rect.color, rect.thickness)

    if with_arrow and f.dist != None:
        y_val = int(h*(1 - f.dist))
        points = np.array([[0, max(y_val - 5, 0)], [0, min(y_val + 5, h)], [10, y_val]], np.int32)
        cv2.fillPoly(image, [points], (0, 0, 255))

    res = Image.fromarray(np.array([[[r, g, b] for [b, g, r] in line] for line in image]))
    res.save(res_path)
    img = face_recognition.load_image_file(res_path)
    enc_arr = face_recognition.face_encodings(img)

    if len(enc_arr) == 0:
        curr_y = 1
    else:
        curr_y = face_recognition.face_distance([face.encoding], enc_arr[0])[0]

    f.dist = curr_y
    
fails = []
distances = []

res_f = open("trained_res_rect/case11/square_fuzzing_slow.txt", "w")

for x in range(num_filters):
    curr_f = create_filter(-1)
    apply_filter(curr_f)
    top_filters_overall.append(curr_f)


for iter in range(training_iters):

    top_filters = []

    # perform 100
    # save top 100
    for _ in range(4*num_filters):
        val = random.randint(0, 10)
        transform = None
        if val < 1: 
            # random filter 10%
            transform = "random"
            curr_f = create_filter(iter)
        elif val < 5:
            # mate 40%
            transform = "mate"
            try:
                [f0, f1] = random.choices(top_filters_overall, weights = weights, k = 2)
                curr_f = mate_filters(f0, f1, iter)
            except:
                curr_f = create_filter(iter)
        else:
            # perturb 50%
            transform = "perturb"
            f = random.choices(top_filters_overall, weights = weights, k = 1)[0]
            try:
                curr_f = perturb_filter(f, iter)
            except:
                curr_f = create_filter(iter)

        apply_filter(curr_f)

        print(f"{iter}: {curr_f.dist}, {transform}")
        res_f.write(f"{iter}\t{curr_f.dist}\t{transform}\n")

        top_filters.append(curr_f)

    top_filters.sort(key = lambda x : (x.dist, -x.iter))
    top_filters_overall = top_filters[-num_filters:]

    top_filter = top_filters_overall[-1]
    apply_filter(top_filter, 
        res_path=f"trained_res_rect/case11/iter{iter}_{top_filter.method}_{top_filter.dist:.3f}.jpg", 
        with_arrow=True)


res_f.close()

