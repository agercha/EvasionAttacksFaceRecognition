import face_recognition
import os.path
from transforms import *
import random

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

# redo to check all duos

genders = ['f', 'm']
glasses = ['n', 'g']
views = ['a', 'b', 'c', 'd', 'e']

def get_lighting_set(sub_num):
    if sub_num < 200:
        return ['q', 'r', 's']
    elif sub_num < 400:
        return ['t', 'u', 'v']
    elif sub_num < 600:
        return ['w', 'x']
    else:
        return ['y', 'z']
    
fails = []
distances = []
basic_faces = dict()

dist_f = open("muct_dist_rotate_fuzz.txt", "w")
fail_f = open("muct_fail_rotate_fuzz.txt", "w")


for sub_num in range(625):
    if sub_num%50 == 0: print(f"on sub num {sub_num}")
    lighting = get_lighting_set(sub_num)[0]
    view = views[0]
    for gender_try in genders:
        for glass_try in glasses:
            curr_path = f"../muct/i{sub_num:03d}{lighting}{view}-{gender_try}{glass_try}.jpg"
            if os.path.exists(curr_path):
                img = face_recognition.load_image_file(curr_path)
                enc_arr = face_recognition.face_encodings(img)
                if len(enc_arr) == 0:
                    face = Face(sub_num, lighting, view, gender_try, glass_try, Transform("none", doNothing), None)
                    fail_f.write(face.get_str())
                else:
                    face = Face(sub_num, lighting, view, gender_try, glass_try, Transform("none", doNothing), enc_arr[0])
                    basic_faces[sub_num] = face

print("done collecting basic faces")

basic_faces_nums = list(basic_faces.keys())
print(basic_faces_nums)

iter = 0
while True:
    if iter%100 == 0: print(f"on iteration {iter}")
    iter += 1
    sub_num = random.choice(basic_faces_nums)
    view = random.choice(views)
    lighting = random.choice(get_lighting_set(sub_num))
    angle = random.randint(0, 360)
    reference_face = basic_faces[sub_num]
    gender = reference_face.gender
    glasses = reference_face.glasses
    curr_path = f"../muct/i{sub_num:03d}{lighting}{view}-{gender_try}{glass_try}.jpg"
    if os.path.exists(curr_path):
        transform = Transform(str(angle), lambda x : rotate(x, angle))
        i = transform.func(curr_path)
        i.save(f"tmp_{transform.name}.jpg")
        img = face_recognition.load_image_file(f"tmp_{transform.name}.jpg")
        enc_arr = face_recognition.face_encodings(img)
        if len(enc_arr) == 0:
            face = Face(sub_num, lighting, view, gender_try, glass_try, transform, None)
            fail_f.write(face.get_str())
            dist = Dist(" ", 1, face, reference_face)
        else:
            face = Face(sub_num, lighting, view, gender_try, glass_try, transform, enc_arr[0])
            reference_face = basic_faces[sub_num]
            dist_val = face_recognition.face_distance([reference_face.encoding], face.encoding)[0]
            dist = Dist(dist_val, " ", face, reference_face)
        dist_f.write(dist.get_str())

dist_f.close()
fail_f.close()
