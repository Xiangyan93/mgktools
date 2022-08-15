import io
import re
import setuptools


with open('mgktools/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

setuptools.setup(
    name='mgktools',
    version=__version__,
    python_requires='>=3.8',
    install_requires=[
        'scikit-learn>=0.24.1',
        'tqdm>=4.62.0',
        'hyperopt>=0.2.5',
        'scipy>=1.6.2',
        'mendeleev>=0.7',
        'rxntools>=0.0.2',
        'pycuda==2020.1',
    ],
    author='Yan Xiang',
    author_email='1993.xiangyan@gmail.com',
    description='Marginalized graph kernel library for molecular property prediction',
    long_description=long_description,
    url='https://github.com/xiangyan93/mgktools',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    data_files=[('hyperparameters', ['mgktools/hyperparameters/additive.json',
                                     'mgktools/hyperparameters/additive-msnorm.json',
                                     'mgktools/hyperparameters/additive-norm.json',
                                     'mgktools/hyperparameters/additive-pnorm.json',
                                     'mgktools/hyperparameters/product.json',
                                     'mgktools/hyperparameters/product-msnorm.json',
                                     'mgktools/hyperparameters/product-norm.json',
                                     'mgktools/hyperparameters/product-pnorm.json'])]
)
