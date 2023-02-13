from keras.preprocessing.image import ImageDataGenerator#, load_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array,array_to_img
import os

# datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )


# Binarization des Images
for img in L:
    imag = Image.open(img)
    pathname, extension = os.path.splitext(img)
    image_name=pathname+"_bin"+".jpg"
    left = 155
    top = 270
    right = 360
    bottom = 1000
    #test = "../datasets/"
    image_name = image_name
    cropped = imag.crop((250,440, 900, 1250)) #moj
    case = (150, 250, 600, 750)
    cropped = imag.crop((1,2, 300, 300)) # meb
    cropped.save(image_name)


L = []
for dirname, _, filenames in os.walk('ok/'):
    for filename in filenames:
        L.append(os.path.join(dirname, filename))

for i in L:
    img = load_img(i)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    datagen.fit(x)
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='ok/', save_prefix='Aug', save_format='jpg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely