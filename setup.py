# =============================================================================#
# Authors: Windsor Nguyen
# File: setup.py
# =============================================================================#

"""Spectral State Space Models."""

import setuptools

setuptools.setup(
    name='spectral_ssm',
    version='1.0',
    description="Dependency manager for Google DeepMind's Spectral State Space Model",
    long_description="""
        Spectral State Space Models. See more details in the
        [`README.md`](https://github.com/windsornguyen/spectral_ssm).
        """,
    long_description_content_type='text/markdown',
    author_email='mn4560@princeton.edu',
    url='https://github.com/windsornguyen/spectral_ssm',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=2.0',
        'uv>=0.2.11'
    ],
    python_requires='>=3.8',
    extras_require={'dev': ['ipykernel>=6.29.4', 'ruff>=0.4.8']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=[
        'pytorch',
        'state space model',
        'spectral filtering',
        'state space model',
        'deep learning',
        'machine learning',
        'time series',
        'dynamical systems',
    ],
    author=
    [
        # In alphabetical order, by last name:
        'Yagiz Devre',
        'Evan Dogariu',
        'Chiara von Gerlach',
        'Isabel Liu',
        'Windsor Nguyen',
        'Dwaipayan Saha',
    ],
)
