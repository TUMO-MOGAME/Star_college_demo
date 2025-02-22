from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function returns a list of package dependencies from requirements.txt.
    """
    requirement_lst: List[str] = []
    
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                requirement = line.strip()
                
                # Ignore empty lines, comments, and '-e .' for editable installs
                if requirement and not requirement.startswith("#") and requirement != "-e .":
                    requirement_lst.append(requirement)
    
    except FileNotFoundError:
        print("Error: 'requirements.txt' file not found.")
    
    return requirement_lst

# Example usage:
print(get_requirements())  # Prints the list of dependencies

setup(
    name="MlProjectWithEmployment",
    version="0.0.1",
    author="Tumo Mogame",
    author_email="tumomogame9@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()  # Ensure this returns a list
)