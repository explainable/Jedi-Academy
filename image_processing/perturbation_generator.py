import numpy as np

class GrayscalePerturbator:
    def __init__(self, image, sample_image=True, grid_dimen=2, stride=2, scale_factor=2, stride_scale=1, cutoff_dimen=None):
        try: 
            self.original = np.array(image, copy=True, dtype="float32")
        except:
            raise Exception("Unable to convert input to NumPy array")

        shape = np.shape(self.original)
        if len(shape) != 2:
            if len(shape) == 1:
                raise Exception("Expecting 2d array of pixels, but got 1d array")

            if len(shape) == 3:
                raise Exception("Expecting 2d array of pixels, but got 3d array")

            raise Exception("Expecting 2d array of pixels, but got " + str(len(shape)) + "d array")
        self.height, self.width = shape

        if (sample_image):
            flat = self.original.flatten() #since mean and var both flatten
            self.mean = np.mean(flat)
            self.variance = np.var(flat)
        else:
            self.mean = 122.5
            self.variance = 30

        self.maskedIndices = []
        self.maskPosition = (0, 0)
        self.done = False
        self.grid_dimen = grid_dimen
        self.stride = stride
        self.scale_factor = scale_factor
        self.cutoff_dimen = cutoff_dimen
        self.stride_scale = stride_scale

    # returns a copy to avoid clients breaking internal logic
    # could return a reference while storing memory of pre-perturbed grid,
    #       but likely not much of a performance improvement, esp for small img
    def nextPerturbation(self):
        if self.done:
            return None

        perturbation = self.applyPerturbation(np.copy(self.original))

        self.updateMask()

        return perturbation

    def __iter__(self):
        return self;

    def __next__(self):
        pert = self.nextPerturbation()
        if (pert is None):
            raise StopIteration()
        return pert

    def applyPerturbation(self, perturbation):
        mask = np.random.normal(loc=self.mean, scale=self.variance, size=(self.grid_dimen, self.grid_dimen))

        maskedIndices = []
        posX, posY = self.maskPosition
        for r, row in enumerate(mask):
            for c, col in enumerate(mask):
                perturbation[posY + r][posX + c] = mask[r][c] 
                maskedIndices.append((posY + r, posX + c))


        #for retrieving indices of masked pixels after calling nextPerturbation
        self.maskedIndices = maskedIndices 

        return perturbation

    def updateMask(self):
        posX, posY = self.maskPosition
        dimen = self.grid_dimen
        stride = self.stride

        xOver = posX + dimen + stride > self.width
        yOver = posY + dimen + stride > self.height

        if xOver and yOver:
            self.maskPosition = (0, 0)
            self.grid_dimen *= self.scale_factor
            self.stride *= self.stride_scale

            if (self.cutoff_dimen):
                if (self.grid_dimen > self.cutoff_dimen):
                    self.done = True
            else: 
                if (self.grid_dimen > self.width / 2 or self.grid_dimen > self.height / 2):
                    self.done = True
        elif xOver:
            self.maskPosition = (0, posY + stride)
        else:
            self.maskPosition = (posX + stride, posY)