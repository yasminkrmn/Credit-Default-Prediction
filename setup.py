from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """
    This function will return list of requirements.

    :param file_path: str
    :return:list
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Yasemin Lheureux',
    author_email = 'gayanekaraman@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )