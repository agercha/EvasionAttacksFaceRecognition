import face_recognition
import os.path

folders = ["faces",
           "centered",
           "rotated",
           "padded",
           "unpadded"]

modifiers = ["centerlight", 
            "glasses", 
            "happy", 
            "leftlight", 
            "noglasses", 
            "normal", 
            "rightlight", 
            "sad", 
            "sleepy", 
            "surprised", 
            "wink"]

endings = [".pgm", ""]

yale_fails = open("yale_fails.txt", "w")

distances = []

for sub_num in range(1,16):
    encodings = []
    for folder in folders:
        for modifier in modifiers:
            for end in endings:
                path = f"../YALE/{folder}/subject{sub_num:02d}.{modifier}{end}"
                if os.path.exists(path):
                    img = face_recognition.load_image_file(path)
                    encoding_arr = face_recognition.face_encodings(img)
                    if len(encoding_arr) == 0:
                        yale_fails.write(f"failed to find face in {path}\n")
                    else:
                        encodings.append((encoding_arr[0], path))
    for (i, (enc1, path1)) in enumerate(encodings):
        for (enc2, path2) in encodings[i:]:
            if not face_recognition.compare_faces([enc1], enc2):
                yale_fails.write(f"failed to recognize {path2} as {path1}\n")
            distances.append([face_recognition.face_distance([enc1], enc2)[0], sub_num, modifier])

yale_fails.close()

print(distances)