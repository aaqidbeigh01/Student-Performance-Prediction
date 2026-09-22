from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines() if req.strip() and req.strip() != "-e ."]
    return requirements

setup(
    name="student_performance",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)