from mnist_cnn import *
from perturbation_generator import *

from keras.datasets import mnist
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
sample_image = (x_train[0])
generator2d = GrayscalePerturbator(sample_image, grid_dimen=1, stride=1, stride_scale=2, cutoff_dimen=1)
model = get_pretrained_mnist_cnn()

if K.image_data_format() == 'channels_first':
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)


target = np.argmax(model.predict(sample_image.reshape(1, 28, 28, 1)))
#print(target)

misses = np.zeros((28, 28), dtype="int32")
for pert in generator2d:
    label = np.argmax(model.predict(pert.reshape(1, 28, 28, 1)))
    if (label != target):
        indices = generator2d.maskedIndices
        #print(indices)
        for index in indices:
            r, c = index
            misses[r][c] += 1

print(misses)

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
i.save("input.png")

j = Image.fromarray(scaled, mode="L")
j.save("heatmap.png")

inverse = 255 - scaled
k = Image.fromarray(inverse, mode="L")
k.save("inverse_heatmap.png")

thresholded = (scaled >= 240) * scaled
l = Image.fromarray(thresholded, mode="L")
l.save("thresholded.png")





