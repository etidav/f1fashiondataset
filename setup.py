import setuptools

install_requires = [
    'pandas == 0.25.1',
    'numpy == 1.21.5',
    'statsmodels == 0.11.1',
    'tqdm == 4.44.1',
    'scikit-learn == 0.22'
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='f1fashiondataset',
    version='0.0.1',
    description='Tool for reproducing the HERMES paper beanchmark results on the F1 fashion dataset',
    long_description= long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/etidav/f1fashiondataset',
    author= 'Etienne DAVID',
    author_email='etienne.david12@gmail.com',
    license='MIT',
    packages=['f1fashiondataset'],
    python_requires='>=3.7',
    install_requires=install_requires
)