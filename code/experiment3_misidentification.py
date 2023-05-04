from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

import face_recognition
import os.path
from transforms import *
import random

import uuid
import copy
import math

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
        self.start_dist = None
        self.target_dist = None
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



num_filters = 50 
training_iters = 500 

weights = [i for i in range(num_filters)] 

start_number = "014"
target_number = "000"

path = f"../muct/i{start_number}qe-fn.jpg"
img=Image.open(path)

start_faces = []
target_faces = []
start_face_encodings = []
target_face_encodings = []

for lighting in ["q", "r", "s"]:
    for position in ["a", "b", "c", "d", "e"]:
        start_path = f"../muct/i{start_number}{lighting}{position}-fn.jpg"
        start_img_rec = face_recognition.load_image_file(start_path)
        w, h = Image.open(start_path).size
        start_enc = face_recognition.face_encodings(start_img_rec)
        start_face = Face(0, "q", "a", "f", "n", Transform("none", doNothing), start_enc[0])
        start_faces.append(start_face)
        start_face_encodings.append(start_enc)

        target_path = f"../muct/i{target_number}{lighting}{position}-fn.jpg"
        target_img_rec = face_recognition.load_image_file(target_path)
        target_enc = face_recognition.face_encodings(target_img_rec)
        target_face = Face(0, "q", "a", "f", "n", Transform("none", doNothing), target_enc[0])
        target_faces.append(target_face)
        target_face_encodings.append(target_enc)

top_filters_overall = []
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
    w_pertub = int(w/25)
    h_pertub = int(h/25)
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
    image = cv2.imread(path)
    for rect in f.rectangles:
        cv2.rectangle(image, rect.start, rect.end, rect.color, rect.thickness)

    if with_arrow and f.dist != None:
        y_start_val = int(h*(1 - f.start_dist))
        points = np.array([[0, max(y_start_val - 5, 0)], [0, min(y_start_val + 5, h)], [10, y_start_val]], np.int32)
        cv2.fillPoly(image, [points], (0, 0, 255))

        y_target_val = int(h*(1 - f.target_dist))
        points = np.array([[w, max(y_target_val - 5, 0)], [w, min(y_target_val + 5, h)], [w - 10, y_target_val]], np.int32)
        cv2.fillPoly(image, [points], (0, 0, 255))

    res = Image.fromarray(np.array([[[r, g, b] for [b, g, r] in line] for line in image]))
    res.save(res_path)
    img = face_recognition.load_image_file(res_path)
    enc_arr = face_recognition.face_encodings(img)

    if len(enc_arr) == 0:
        return 0
    else:
        start_y_arr = face_recognition.face_distance(start_face_encodings, enc_arr[0])[0]
        target_y_arr = face_recognition.face_distance(target_face_encodings, enc_arr[0])[0]
        start_y = sum(start_y_arr)/15
        target_y = sum(target_y_arr)/15
        curr_y = start_y - target_y

    if not with_arrow: 
        f.start_dist = start_y
        f.target_dist = target_y
        f.dist = curr_y
    
    return 1
    
fails = []
distances = []

res_f = open("train_res_trick/case6/square_fuzzing_slow.txt", "w")


filter_count = 0
while filter_count < num_filters:
    curr_f = create_filter(-1)
    filter_count += apply_filter(curr_f)
    top_filters_overall.append(curr_f)


for iter in range(training_iters):

    top_filters = []

    # perform 100
    # save top 100
    filter_count = 0
    while filter_count < 2*num_filters:
        val = random.randint(0, 10)
        transform = None
        if val < 2: 
            # random filter 20%
            transform = "random"
            curr_f = create_filter(iter)
        elif val < 5:
            # mate 30%
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

        if apply_filter(curr_f) == 1:
            filter_count += 1

            print(f"{iter}: {curr_f.start_dist}, {curr_f.target_dist}, {transform}")
            res_f.write(f"{iter}\t{curr_f.start_dist}\t{curr_f.target_dist}\t{transform}\n")

            top_filters.append(curr_f)

    top_filters += top_filters_overall[-int(num_filters/10):] # keep top five so far of all time
    if iter%2 == 0: # 50% of time
        func = lambda x : x.dist
    elif iter%4 == 1: # 25% of time
        func = lambda x : x.start_dist
    else: # 25% of time
        func = lambda x : -x.target_dist

    top_filters.sort(key = func)
    
    top_filters_overall = top_filters[-num_filters:]

    top_filter = top_filters_overall[-1]
    apply_filter(top_filter, 
        res_path=f"train_res_trick/case6/iter{iter}_{top_filter.start_dist:.3f}_{top_filter.target_dist:.3f}_{top_filter.method}.jpg", 
        with_arrow=True)


res_f.close()

