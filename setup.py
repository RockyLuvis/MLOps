from setuptools import find_packages,setup # type: ignore
from typing import List

HYPEN_E_DOT='-e .'

#def get_requirements(file_path:str)->List[str]:
#    requirements=[]
#    with open(file_path) as file_obj:
#        requirements=file_obj.readlines()
#        requirements=[req.replace("\n","") for req in requirements]
#
#        if HYPEN_E_DOT in requirements:
#            requirements.remove(HYPEN_E_DOT)
#    return requirements

setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Raveendra Seetharam',
    author_email='ravibhattinkw@gmail.com',
    description="A python package for Predicting price of diamond",
    #long_description="long_description",
    long_description_content="text/markdown",
    url=f"https://github.com/RockyLuvis/MLOps",
    project_urls={
        "Bug_Tracker": f"https://github.com/RockyLuvis/MLOps/issues",
    },
    package_dir={"": "src"},

    install_requires=["scikit-learn","pandas","numpy","pytest","mypy"],
    #install_requires=get_requirements("requirements_dev.txt"),
    packages=find_packages(where = "src")
)
