import os

# clone the pints repo
if os.path.isfile('./pints'):
    pass
else:
    os.system('git clone https://github.com/pints-team/pints.git')

# install dependencies
os.system('pip install pints/')