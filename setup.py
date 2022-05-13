# Copyright (c) 2021-2022 Javad Komijani


from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


packages = [
        'normflow',
        'normflow.lib',
        'normflow.util',
        'normflow.models',
        'normflow.measure'
        ]

package_dir = {
        'normflow': 'src',
        'normflow.lib': 'src/lib',
        'normflow.util': 'src/util',
        'normflow.models': 'src/models',
        'normflow.measure': 'src/measure'
        }

setup(name='normflow',
      version='1.0',
      description='Normalizing flows as generative models for QFT simulations',
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/normflow',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      install_requires=['numpy>=1.1', 'torch>=1.1'],
      zip_safe=False
      )
