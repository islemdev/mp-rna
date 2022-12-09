from tensorflow import keras
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
from pretty_confusion_matrix import pp_matrix_from_data

model = keras.models.load_model('mammo_model3')
def calc_perf(true_class, pred_class):
    if true_class["CAN"] != 0:
        print("CAN = "+str(pred_class["CAN"]/true_class["CAN"] ))
    if true_class["BEN"] != 0:
        print("BEN = "+str(pred_class["BEN"]/true_class["BEN"] ))
    if true_class["NOR"] != 0:
        print("NOR = "+str(pred_class["NOR"]/true_class["NOR"] ))

#type = "BEN"
dest = join("ddsm1", "ddsmROI")
onlyfiles = [f for f in listdir(dest) if isfile(join(dest, f))]
n = 0
true_n = 0
nor = 0
ben = 0
can = 0
print(len(onlyfiles))
class_names = ["BEN", "CAN", "NOR"]
classes = {"BEN":0, "CAN":0, "NOR":0}
true_classes ={"BEN":0, "CAN":0, "NOR":0}
pred = []
ytrue = []
for f in onlyfiles:
    try:
        n += 1
        print(n)
        ext = f.split('.')[1]
        if ext == 'txt':
            continue
        file_name = int(f.split('.')[0])
        #print(file_name)
        type = ""
        if file_name >= 0 and file_name <= 9215:
            type="NOR"
            nor += 1
        elif file_name >= 9216 and file_name <= 10103:
            type="BEN"
            ben += 1
        elif file_name >= 10104 and file_name <= 11218:
            type ="CAN"
            can += 1
        else:
            continue
        img = keras.preprocessing.image.load_img(
            join(dest, f), target_size=(128, 128),
            color_mode="grayscale"
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        true_classes[type] +=1
        ytrue.append(type)
        predictions = model.predict(img_array, verbose=0)
        index = predictions[0].argmax()
        classes[class_names[index]] += 1
        pred.append(class_names[index])

        #calc_perf(true_classes, classes)
    except:
        print("damaged image")



calc_perf(true_classes, classes)
print(true_classes)
print(classes)

ypred=np.array(pred)
ytrue=np.array(ytrue)

pp_matrix_from_data(ytrue, ypred)