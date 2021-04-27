
from setuptools import setup, find_packages

setup(
      name='sinnfull',
      version='0.1.0dev',
      description="",
      python_requires=">=3.8",

      author="Alexandre René",
      author_email="",

      license='MPL',

      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only'
      ],

      packages=find_packages(),

      install_requires=[  # see also env.yaml
          # Packages installed from repositories
          'parameters',
          'pymc3',
          'sumatra>=0.8dev',
          'theano_shim>=0.3.0',
          'mackelab_toolbox[iotools,typing,utils,parameters]>=0.2.0a1',
          'smttask>=0.2.0b1',
          'sinn>=0.2.0rc1',
      ]

 )
