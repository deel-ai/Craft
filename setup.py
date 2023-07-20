from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Craft-xai",
    version="0.0.3",
    description="Automatic Concept Extraction with CRAFT",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas FEL, Agustin PICARD, Louis Béthune, Thibaut BOISSIN",
    author_email="thomas_fel@brown.edu",
    license="MIT",
    install_requires=['numpy', 'scikit-learn', 'scikit-image',
                      'matplotlib', 'scipy', 'opencv-python'],
    extras_require={
        "tests": ["pytest", "pylint"],
        "docs": ["mkdocs", "mkdocs-material", "numkdoc"],
    },
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)