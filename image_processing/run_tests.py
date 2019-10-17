from perturbation_generator import *

# Begin 2D Array Initializer Tests #
try:
    generator2d = GrayscalePerturbator([1, 2, 3, 4, 5])
except Exception as e: 
    if str(e) != "Expecting 2d array of pixels, but got 1d array":
        print("1d array test failed")
    else:
        print("1d array test passed")


generator2d = GrayscalePerturbator([[1,2,3,4,5], [1, 2, 3, 4, 5]])
print("2d array test passed")

try:
    generator2d = GrayscalePerturbator([[[1,2,3,4,5], [1, 2, 3, 4, 5]], [[1,2,3,4,5], [1, 2, 3, 4, 5]]])
except Exception as e: 
    if str(e) != "Expecting 2d array of pixels, but got 3d array":
        print("3d array test failed")
    else:
        print("3d array test passed")

try:
    generator2d = GrayscalePerturbator([[[[1,2,3,4,5], [1, 2, 3, 4, 5]], [[1,2,3,4,5], [1, 2, 3, 4, 5]]],
                                        [[[1,2,3,4,5], [1, 2, 3, 4, 5]], [[1,2,3,4,5], [1, 2, 3, 4, 5]]]])
except Exception as e: 
    if str(e) != "Expecting 2d array of pixels, but got 4d array":
        print("4d array test failed")
    else:
        print("4d array test passed")
# End 2D Array Initializer Tests #