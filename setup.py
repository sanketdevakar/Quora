from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->list[str]:
    '''
    This function will return the list of libraries to be installed for this project
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.read()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name= "Quora Kaggle Solution",
    version="0.0.1",
    author="Sanket",
    author_email="sanketdevakar09@gmail.com",
    packages=find_packages(),
    install_packages= get_requirements('requirements.txt')

)