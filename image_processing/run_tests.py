from perturbation_generator import *

# Begin 2D Array Initializer Tests #
try:
    generator2d = GrayscalePerturbator([1, 2, 3, 4, 5])
except Exception as e: 
    if str(e) != "Expecting 2d array of pixels, but got 1d array":
        print("1d array test failed")
    else:
        print("1d array test passed")

try: 
    generator2d = GrayscalePerturbator([[1,2,3,4,5], [1, 2, 3, 4, 5]])
    print("2d array test passed")
except Exception as e: 
    print("Valid input threw exception for 2d array initializer: ")
    print(str(e))

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



# Begin 2D Array Perturbation Tests #
try:
    generator2d = GrayscalePerturbator([[1, 0, 1, 0], 
                                        [0, 1, 0, 1],
                                        [1, 0, 1, 0],
                                        [0, 1, 0, 1]])

    perturbed = generator2d.nextPerturbation()
    while(perturbed is not None):
        perturbed = generator2d.nextPerturbation()
    print("Default perturbation passed")
except Exception as e: 
    print("Valid input threw exception for default perturbation: ")
    print(str(e))

try:
    generator2d = GrayscalePerturbator([[1, 0, 1, 0], 
                                        [0, 1, 0, 1],
                                        [1, 0, 1, 0],
                                        [0, 1, 0, 1]], grid_dimen=1, stride=1)

    perturbed = generator2d.nextPerturbation()
    while(perturbed is not None):
        perturbed = generator2d.nextPerturbation()
    print("Scaling perturbation passed")
except Exception as e: 
    print("Valid input threw exception for scaling perturbation: ")
    print(str(e))

try:
    generator2d = GrayscalePerturbator([[1, 0, 1, 0], 
                                        [0, 1, 0, 1],
                                        [1, 0, 1, 0],
                                        [1, 0, 1, 0],
                                        [1, 0, 1, 0]], grid_dimen=1, stride=1)

    perturbed = generator2d.nextPerturbation()
    while(perturbed is not None):
        perturbed = generator2d.nextPerturbation()
    print("Non-square perturbation passed")
except Exception as e: 
    print("Valid input threw exception for Non-square perturbation: ")
    print(str(e))

try:
    for pert in GrayscalePerturbator([[1, 0, 1, 0], 
                                            [0, 1, 0, 1],
                                            [1, 0, 1, 0],
                                            [1, 0, 1, 0],
                                            [1, 0, 1, 0]], grid_dimen=1, stride=1):
        pass
    print("Perturbation generator syntax passed")
except Exception as e: 
    print("Valid input threw exception for generator syntax: ")
    print(str(e))
# End 2D Array Perturbation Tests #