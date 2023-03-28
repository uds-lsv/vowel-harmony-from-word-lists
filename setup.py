from setuptools import setup, find_packages

setup(
    name='eff',
    description='MT',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/digling/eff.git',
    author='Julius Steuer',
    author_email='julius.steuer@mailbox.org',
    license='GPL-v3',
    version="0.1.0.dev0",
    packages=find_packages(
        where="src",
        exclude=[]
        ),
    package_dir={'':'src'},
    zip_safe=False,
    install_requires = [
        #"enum34",
        "pyclts",
        "numpy",
        # "torch",
        "cltoolkit",
        "dill",
        "lingpy",
        "anytree",
        "pandas",
        "scipy",
        "matplotlib",
        "sklearn"
    ],
    extras_require={
        'dev': ['flake8', 'wheel', 'twine'],
        'test': [
            'pytest>=6',
            'pytest-mock',
            'pytest-cov',
            'coverage',
        ],
    }
)
