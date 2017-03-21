import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "speech-recognition",
    version = "0.0.1",
    author = "Emre Akkaya",
    author_email = "emrekaganakkaya@gmail.com",
    description = ("Speech recognition course material"),
    license = "LGPLv3",
    keywords = "speech recognition tutorial",
    packages=['speech-recognition', ],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Utilities",
        "License :: LGPLv3",
    ],
)