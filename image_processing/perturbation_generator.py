import numpy as np

class GrayscalePerturbator:
    def __init__(self, image):
        try: 
            self.original = np.array(image, copy=True)
        except:
            raise Exception("Unable to convert input to NumPy array")

        shape = np.shape(self.original)
        if len(shape) != 2:
            if len(shape) == 1:
                raise Exception("Expecting 2d array of pixels, but got 1d array")

            if len(shape) == 3 and shape[2] != 1:
                raise Exception("Expecting 2d array of pixels, but got 3d array")

            raise Exception("Expecting 2d array of pixels, but got " + str(len(shape)) + "d array")


        self.currentIteration = 0
        self.done = False

    def nextPerturbation(self):
        if self.done:
            return None

        perturbation = np.copy(self.original)

        # apply perturbation

        self.currentIteration += 1

        # check if done, update self.done accordingly

        return perturbation