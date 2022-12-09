import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from tensorflow import Tensor
import matplotlib as plt
from pretty_confusion_matrix import pp_matrix_from_data


image_size = (128, 128)
batch_size = 128

# Create the training dataset with data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

train_ds = datagen.flow_from_directory(
    "mammoimages",
    subset="training",
    seed=123,
    target_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

# Create the validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "mammoimages",
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
    labels='inferred',
    label_mode="categorical"
)

classes=val_ds.class_names # ordered list of class names
print(classes)
#print(val_ds)
ytrue=[]



model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(128, 128, 1) ),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),

    layers.Dense(3, activation="softmax"),

])

model.summary()
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer="adam"
)
epochs = 50

model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)

print(model.evaluate(val_ds))

model.save("mammo_model")
#class_names = {"BEN": 0, "CAN": 1, "NOR": 2}

sns.set_style('darkgrid')
classes=val_ds.class_names # ordered list of class names
#print(val_ds)
ytrue=[]


for images, label in val_ds:
    for e in label:
        for index, i in enumerate(e.numpy()):
            if i == 1:
                ytrue.append(classes[index]) # list of class names associated with each image file in test dataset
ypred=[]
errors=0
count=0
preds=model.predict(val_ds, verbose=1) # predict on the test data
for i, p in enumerate(preds):
    count +=1
    index=np.argmax(p) # get index of prediction with highest probability
    klass=classes[index]
    ypred.append(klass)
    if klass != ytrue[i]:
        errors += 1
acc= (count-errors)* 100/count
msg=f'there were {count-errors} correct predictions in {count} tests for an accuracy of {acc:6.2f} % '
print(msg)
ypred=np.array(ypred)
ytrue=np.array(ytrue)

pp_matrix_from_data(ytrue, ypred)

clr = classification_report(ytrue, ypred, target_names=classes)
print("Classification Report:\n----------------------\n", clr)