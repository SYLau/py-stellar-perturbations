# workdir is a variable that contains the path to the working directory (directory containing src and test)
# Can use relative path or absolute path
# For each project that uses self-defined modules, it is important to add workdir to sys.path (see examples in test/poly_MR.ipynb)
import os
# This line is equivalent to:
# workdir = '../../'
workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir,os.pardir)