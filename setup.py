from setuptools import setup, find_packages

setup(
  name = 'block-recurrent-transformer',
  packages = find_packages(exclude=['examples']),
  version = '0.1.0',
  license='MIT',
  description = 'Block Recurrent Transformers, an implementation of [Hutchins & Schlag et al.](https://arxiv.org/abs/2203.07852v1)',
  author = 'Dashiell Stander',
  author_email = 'dash.stander@gmail.com',
  url = 'https://github.com/dashstander/block-recurrent-transformer',
  keywords = [
    'machine learning',
    'attention',
    'transformers',
    'long range transformer'
  ],
  install_requires=[
    'torch>=1.9',
    'einops>=0.3',
    'entmax',
    'x-transformers'
  ]
)

