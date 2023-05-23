from setuptools import setup, find_packages


setup(
    name='mk_input_db',
    packages=find_packages(where='src'),
    package_data={
        "mk_input_db.package_data": ["*"],
    },
    entry_points={
        'console_scripts': [
            'mk_input_db = mk_input_db:run',
        ]
    },
    version='0.1',
    license='MIT',
    description = 'My package description',
    description_file = "README.md",
    author="Julien Braine",
    author_email='julienbraine@yahoo.fr',
    url='https://github.com/JulienBrn/mk_input_db',
    download_url = 'https://github.com/JulienBrn/mk_input_db.git',
    package_dir={'': 'src'},
    keywords=['python'],
    install_requires=[],
    #['pandas', 'matplotlib', 'PyQt5', "sklearn", "scikit-learn", "scipy", "numpy", "tqdm", "beautifullogger", "statsmodels", "mat73", "psutil"],
    python_requires=">=3.10"
)
