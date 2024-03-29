import os
import distutils
from setuptools import setup, find_packages, Command
from setuptools.command.build_py import build_py

# Load text for description and license
with open('README.md') as f:
    readme = f.read()


class BuildPyCommand(build_py):
  """Custom build command."""

  def run(self):
    self.run_command('pints')
    build_py.run(self)


class PINTS_installer(Command):
    """A costum command to install PINTS in the process of setting up the PKPD module.
    """
    description = ('Clones and pip installs PINTS.')

    def initialize_options(self) -> None:
        self.commands = []

    def finalize_options(self) -> None:
        self.commands = [
            'git clone https://github.com/pints-team/pints.git',
            'pip install pints/']

    def run(self):
        """Run commands.
        """
        git_clone_cmd = self.commands[0]
        pip_install_cmd = self.commands[1]

        # clone PINTS repository.
        if os.path.isdir('pints'):
            self.announce(
                'PINTS repository exists in directory',
                level=distutils.log.INFO)
        elif not os.path.isfile('./pints'):
            self.announce(
                'Running command: %s' % str(git_clone_cmd),
                level=distutils.log.INFO)
            os.system(git_clone_cmd)

        # pip install PINTS
        os.system(pip_install_cmd)


# Go!
setup(
    # Module name (lowercase)
    name='Gray-Scott-PDE',
    version='0.2dev',

    # Description
    description='PDE solver and parameter inference for the Gray-Scott model.',
    long_description=readme,

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    url='https://github.com/sabs-r3-projects/Gray-Scott-PDE',

    # Packages to include
    packages=find_packages(include=('grayscott',)),

    # customised commands
    cmdclass={
        'pints': PINTS_installer,
        'build_py': BuildPyCommand
    },

    # List of dependencies
    install_requires=[
        'cma>=2',
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.5',
        'tabulate',
        'pytest',
        'jupyter',
        'tqdm'
    ]
)
