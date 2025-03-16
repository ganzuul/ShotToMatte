from setuptools import setup, find_packages

setup(
    name="shottomatte",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "kornia>=0.6.0",
    ],
    extras_require={
        "full": [
            "detectron2",  # For instance segmentation
            "cupy-cuda111",  # For CUDA acceleration
        ]
    },
    entry_points={
        "console_scripts": [
            "shottomatte=shottomatte.cli:main",
        ],
    },
    author="AI Vision Labs",
    author_email="info@aivisionlabs.com",
    description="Extract matte paintings from animated content using deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aivisionlabs/shottomatte",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 