import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages(where="src")

setuptools.setup(
    name='trex_python',
    version='0.1.2',    
    description='A Python implementation of the T-ReX algorithm',
    url='https://git.ias.u-psud.fr/tbonnair/t-rex.git',
    author='Tony Bonnaire',
    author_email='bonnaire.tonyim@gmail.com',
    license='GNU General Public License v3.0',
    package_dir={"": "src"},
    packages=pkgs,
    python_requires=">=3.7",
)