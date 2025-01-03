from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="HighChromaticNumberGraphs",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        "console_scripts": [
            "my_project=src.main:main",
        ],
    },
)