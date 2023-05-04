import face_recognition
import os.path
from transforms import *
from datetime import datetime

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
    def __init__(self, dist, face1, face2):
        self.dist = dist
        self.face1 = face1
        self.face2 = face2

    def __repr__(self):
        return f"{self.dist}\t{self.face1}{self.face2}\n"
    
    def get_str(self):
        return f"{self.dist}\t{self.face1}{self.face2}\n"
    
def rotate45(img_path):
    return rotate(img_path, 45)

def rotate90(img_path):
    return rotate(img_path, 90)

def flip(img_path):
    return rotate(img_path, 180)

def blurWeak(img_path):
    return blur(img_path, 5)

def blurStrong(img_path):
    return blur(img_path, 15)

def doNothing(img_path):
    return Image.open(img_path)

def contrast(img_path):
    return adjust(img_path, contrast=2)

def lighten(img_path):
    return adjust(img_path, brightness=50)

def darken(img_path):
    return adjust(img_path, brightness=-50)

def darkenEyes(img_path):
    return darken_eyes(img_path, 15)

def darkenShadowsSlow(img_path):
    return darken_shadows(img_path, 15)

def darkenShadowsFast(img_path):
    return darken_shadows_fast(img_path, opacity=0.75, radius=40)

def clown(img_path):
    return clown_makeup(img_path, opacity = 0.2, radius = 30)

class Transform():
    def __init__(self, name, func):
        self.name = name
        self.func = func

transforms = [
    Transform("rotate45", rotate45),
    Transform("rotate90", rotate90),
    Transform("flip", flip),
    Transform("blurWeak", blurWeak),
    Transform("blurStrong", blurStrong),
    Transform("contrast", contrast),
    Transform("lighten", lighten),
    Transform("darken", darken),
    Transform("darkenEyes", darkenEyes),
    Transform("darkenShadowsSlow", darkenShadowsSlow),
    Transform("darkenShadowsFast", darkenShadowsFast),
    Transform("clown", clown),
    Transform("none", doNothing),
    Transform("random1", add_random),
    Transform("none", doNothing),
    Transform("random2", add_random2)
]

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

dist_f = open(f"muct_distances_slow_transforms_all.txt", "a")
fail_f = open(f"muct_fails_slow_transforms_all.txt", "a")

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
while True:
    sub_num = random.choice(basic_faces_nums)
    reference_face = basic_faces[sub_num]
    gender = reference_face.gender
    glasses = reference_face.glasses
    lighting = random.choice(get_lighting_set(sub_num))
    view = random.choice(views[0])
    curr_path = f"../muct/i{sub_num:03d}{lighting}{view}-{gender}{glasses}.jpg"
    if os.path.exists(curr_path):
        transform = random.choice(transforms)
        # for transform in transforms:
        print(transform.name, curr_path)
        i = transform.func(curr_path)
        i.save(f"tmp_{transform.name}.jpg")
        img = face_recognition.load_image_file(f"tmp_{transform.name}.jpg")
        enc_arr = face_recognition.face_encodings(img)
        if len(enc_arr) == 0:
            face = Face(sub_num, lighting, view, gender_try, glass_try, transform, None)
            fail_f.write(face.get_str())
        else:
            face = Face(sub_num, lighting, view, gender_try, glass_try, transform, enc_arr[0])
            dist_val = face_recognition.face_distance([reference_face.encoding], face.encoding)[0]
            dist = Dist(dist_val, face, reference_face)
            dist_f.write(dist.get_str())

dist_f.close()
fail_f.close()
