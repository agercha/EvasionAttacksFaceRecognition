import face_recognition
import os.path

# redo to check all duos

genders = ['f', 'm']
glasses = ['n', 'g']
views = ['a', 'b', 'c', 'd', 'e']


muct_fails = open("muct_fails.txt", "w")
muct_distances = open("muct_distances.txt", "w")

def get_lighting_set(sub_num):
    if sub_num < 200:
        return ['q', 'r', 's']
    elif sub_num < 400:
        return ['t', 'u', 'v']
    elif sub_num < 600:
        return ['w', 'x']
    else:
        return ['y', 'z']
    
class SubjectImg():
    def __init__(self, encoding, subject, gender, glasses, lighting, view):
        self.encoding = encoding
        self.subject = subject
        self.gender = gender
        self.glasses = glasses
        self.lighting = lighting 
        self.view = view
    
    def get_str(self):
        return f"{self.subject}\t{self.gender}\t{self.glasses}\t{self.lighting}\t{self.view}"

for sub_num in range(625):
    if sub_num%5 == 0: print(f"on subject {sub_num}")
    encodings = []
    lightings = get_lighting_set(sub_num)
    for gender_try in genders:
        for glass_try in glasses:
            for lighting in lightings:
                for view in views:
                    curr_path = f"../muct/i{sub_num:03d}{lighting}{view}-{gender_try}{glass_try}.jpg"
                    if os.path.exists(curr_path):
                        img = face_recognition.load_image_file(curr_path)
                        enc_arr = face_recognition.face_encodings(img)
                        if len(enc_arr) == 0:
                            muct_fails.write(f"failed to find face in {curr_path}\n")
                        else:
                            enc = enc_arr[0]
                            sub_img = SubjectImg(encoding=enc, subject=sub_num, gender=gender_try, glasses=glass_try, lighting=lighting, view=view)
                            encodings.append(sub_img)

    if len(encodings) > 0:
        plain_enc = encodings[0]
        for test_enc in encodings[1:]:
            d = face_recognition.face_distance([test_enc.encoding], plain_enc.encoding)[0]
            muct_distances.write(f"{d}\t{test_enc.get_str()}\t{plain_enc.get_str()}\n")

muct_distances.close()