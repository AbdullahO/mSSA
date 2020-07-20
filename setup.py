from setuptools import setup, find_packages

setup(name='mSSA',
      version='0.1',
      description='Multivariate Singular Spectrum Analysis (mSSA): Forecasting and Imputation algorithm for multivariate'
                  ' time series.',
      author='Abdullah Alomar',
      license='Apache 2.0',
      packages=['mssa', 'mssa.examples', 'mssa.examples.testdata', 'mssa.examples.testdata.tables', 'mssa.src', 'mssa.src.algorithms','mssa.src.prediction_models' ],
      install_requires=['numpy','h5py', 'pandas','sklearn','scipy'],
      zip_safe=False,
      include_package_data=True,
      package_data={'mSSA': ['example/testdata/tables/*.csv']})
