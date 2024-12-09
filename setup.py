from setuptools import setup

with open('./requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Nd-Conv',
    version='1.0',
    packages=['src', 'src.nd_conv', 'src.utils'],
    url='https://github.com/willxxy/Nd-Conv',
    license='MIT',
    author='William Jongwon Han',
    author_email='wjhan@andrew.cmu.edu',
    description='Open source code of Nd-Conv',
    install_requires=required,
)