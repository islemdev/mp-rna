from pathlib import Path
import imghdr
import os
from os import listdir
from os.path import isfile, join
import shutil



def check(dir):
    data_dir = "./mammoimages/"+dir
    image_extensions = [".png", ".jpg"]  # add there all your images file extensions

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                os.remove(filepath)

            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

dirs = ["CAN", "NOR", "BEN"]
for d in dirs:
    print("dir is "+d)
    #check(d)

onlyfiles = [f for f in listdir("./train") if isfile(join("./train", f))]
can = 0
nor = 0
ben = 0
for f in onlyfiles:
    ext = f.split('.')[1]
    if ext == 'txt':
        continue
    file_name = int(f.split('.')[0])
    print(file_name)
    type=""
    if file_name >= 0 and file_name <= 9215 and nor < 400:
        type="NOR"  # NOR
        nor +=1
    elif file_name >= 9216 and file_name <= 10103 and ben < 300:
        type = "BEN"  # BEN
        ben+=1
    elif file_name >= 10104 and file_name <= 11218 and can < 300:
        type = "CAN" # CAN
        can+=1
    else:
        break
    dest = join("mammoimages", type)
    shutil.move("./train/"+f, join(dest, f))
print(nor, ben, can)
