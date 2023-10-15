from setuptools import setup, find_packages

setup(
    name="UniTrain",
    version="0.2.4",
    author="Ishan Upadhyay",
    author_email="ishan.upadhyay.iitb@gmail.com",
    description="A generalized training framework for Deep Learning Tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ishan121028/UniTrain",
    download_url="https://github.com/ishan121028/UniTrain/archive/v0.1.tar.gz",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "torchsummary",
        "opencv-python",
    ],
    keywords=["Deep Learning", "Machine Learning", "Training Framework"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
