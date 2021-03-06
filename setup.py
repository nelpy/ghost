from setuptools import setup, find_packages
import io

from distutils.util import convert_path

vinfo = {}
ver_path = convert_path('ghost/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), vinfo)

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

# long_description = read('README.rst')

setup(
    name='ghost',
    version=vinfo['__version__'],
    # url='https://github.com/nelpy/ghost/',
    # download_url = 'https://github.com/nelpy/ghost/' + main_ns['__version__'],
    license='MIT License',
    author='Joshua Chu',
    install_requires=['numpy>=1.15.0',
                      'scipy>=1.6.1', # for scipy.fft
                      'matplotlib>=3.0',
                     ],
    author_email='jpc6@rice.edu',
    description='Numpy and nelpy compatible spectral analysis toolbox',
    # long_description=long_description,
    packages=find_packages(),
    keywords = "neuroscience spectral analysis",
    include_package_data=True,
    platforms='any'
)
