from setuptools import setup, find_packages


setup(
    name = "first_break_picking",
    version = "0.0.1",
    description = "A package to perform first break",
    long_description = open("readme.md").read(),
    author='Amir Mardan',
    uthor_email = "mardan.amir.h@gmail.com",
    license = "MIT",
    packages=find_packages(exclude=["*.pyc"]),
    include_package_data=True,
    package_data={
        "": ["*.txt", "fb_20.tar"],
        },
    install_requires = [
        # "segyio",
        "pandas",
        "openpyxl",
        "numpy",
        "torch",
        "torchvision",
        "tqdm",
        "matplotlib",
        "segmentation_models_pytorch"
                        ],
    zip_safe = False
)