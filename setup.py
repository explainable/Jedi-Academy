
from setuptools import setup

DESC = 'This is a module to do ML stuff' # TODO: Seb/Aidan make this not bad

setup(  name='Jedi-Academy-Explainable',
        version='0.1',
        description=DESC,
        url='http://github.com/explainable/Jedi-Academy',
        author='Explainable',
        author_email='explainable@outlook.com',
        #liscense='MIT', # TODO: Fill this is when we decide on what Liscense
        packages=['image_processing', 'tabular'],
        package_dir={'image_processing': 'image_processing'},
        package_data={'image_processing': ['data/*.h5'], 'tabular': ['data/*.csv']},
        include_package_data=True,
        install_requires=[
            'tensorflow',
            'keras',
            'sklearn',
            'pandas',
            'pillow',
        ],
        zip_safe=False # Not sure if this should be true or false, false seemed safer
)
