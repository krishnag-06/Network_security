from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirement_lst:List[str] = []
    try:
        with open("requirements.txt", "r") as files:
            lines  = files.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")

    return requirement_lst

setup(
    name = "Network security",
    version = "0.0.1",
    author = "Krishna Goyal",
    author_email= "krishnagoyal06@icloud.com",
    packages= find_packages(),
    install_requires = get_requirements()
)