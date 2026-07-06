from setuptools import find_packages, setup

setup(
    name="adversarial-attacks-research",
    version="0.1.0",
    description="White-box adversarial attacks (FGSM, PGD, C&W) on MNIST and CIFAR-10",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tabulate",
    ],
)
