from setuptools import setup
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='dapper',
    packages=['dapper'],
    version='0.1-develop',
    description='Dataset Analysis, Processing, and Presentaton of Experimental Results',
    author='Jeremy McGibbon',
    author_email='mcgibbon@uw.edu',
    install_requires=reqs,
#    url='https://github.com/mcgibbon/atmos',
    keywords=['atmospheric', 'geoscience', 'science', 'netcdf'],
    classifiers=[],
    license='MIT',
)
