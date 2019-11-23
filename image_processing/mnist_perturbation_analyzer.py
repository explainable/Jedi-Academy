from mnist_cnn import *
from perturbation_generator import *

from keras.datasets import mnist
from PIL import Image

from sys import argv 

sample = 0
if len(argv) == 1:
    print("No filename provided, using default")
    filename = ""
    
else:
    filename = argv[1] + "_"
    if len(argv) > 2:
        sample = int(argv[2])

ADVERSARIAL = False

NUMBER_TO_SAMPLE_INDEX = {
    0: 1,
    1: 6,
    2: 5,
    3: 7,
    4: 2,
    5: 0,
    6: 13,
    7: 15,
    8: 17,
    9: 4
}
sampleIndex = NUMBER_TO_SAMPLE_INDEX[sample]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
sample_image = (x_train[sampleIndex]) # by 17, all 10 digits are represented at least once
generator2d = GrayscalePerturbator(sample_image, grid_dimen=1, stride=1, stride_scale=1, cutoff_dimen=28, is_adversarial=ADVERSARIAL)
model = get_pretrained_mnist_cnn()

if K.image_data_format() == 'channels_first':
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)

target = y_train[sampleIndex]

misses = np.zeros((28, 28), dtype="float32")
for pert in generator2d:
    label = np.argmax(model.predict(pert.reshape(1, 28, 28, 1)))
    if ((not ADVERSARIAL) and label == target) or (ADVERSARIAL and label != target):
        indices = generator2d.maskedIndices
        for index in indices:
            r, c, grid_dimen = index
            misses[r][c] += 1.0 / (grid_dimen ** 2)
    else:
        pass#print("Failed to match " + str(target) + " as " + str(label))

maxMiss = np.max(misses)
if maxMiss != 0:
    scaled = np.array(misses * (255.0 / maxMiss), dtype="uint8")
else:
    scaled = np.array(misses, dtype="uint8")

#for row in sample_image:
#    print(np.array2string(row, max_line_width=np.inf))
#print()
#for row in scaled:
#    print(np.array2string(row, max_line_width=np.inf))

i = Image.fromarray(sample_image, mode="L")
i.save(filename + "input.png")

j = Image.fromarray(scaled, mode="L")
j.save(filename + "heatmap.png")

inverse = 255 - scaled
k = Image.fromarray(inverse, mode="L")
k.save(filename + "inverse_heatmap.png")

thresholded = (scaled >= 240) * scaled
l = Image.fromarray(thresholded, mode="L")
l.save(filename + "thresholded.png")





