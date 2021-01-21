from distutils.core import setup
  

setup(name='data-learning-boilerplate',
      version='1.0.0',
      install_requires=[
            'hydra-core',
            'torch',
            'wandb',
            'pytorch_lightning',
            'async-savers',
            'tqdm'
      ],
      description='A boilerplate for data generating and learning projects',
      author='Jacky Liang',
      author_email='jackyliang@cmu.edu',
      packages=['data_learning_boilerplate']
     )
