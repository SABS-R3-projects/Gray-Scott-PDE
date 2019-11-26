from setuptools import setup, find_packages

# Load text for description and license
with open('README.md') as f:
    readme = f.read()


# Go!
setup(
    # Module name (lowercase)
    name='Gray-Scott-PDE',
    version='0.1dev',

    # Description
    description='PDE solver and parameter inference for the Gray-Scott model.',
    long_description=readme,

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    url='https://github.com/sabs-r3-projects/Gray-Scott-PDE',

    # Packages to include
    packages=find_packages(include=('grayscott',)),

    # install PINTS
    scripts=['install_pints.py'],

    # List of dependencies
    install_requires=[
        'cma>=2',
        'numpy>=1.8',
        'scipy>=0.14',
        # Note: Matplotlib is loaded for debug plots, but to ensure pints runs
        # on systems without an attached display, it should never be imported
        # outside of plot() methods.
        'matplotlib>=1.5',
        'tabulate',
    ]
)