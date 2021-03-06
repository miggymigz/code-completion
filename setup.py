import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='code-completion',
    version='0.0.1',
    author='Miguel Victor Remulta',
    author_email='miguelvictor.remulta@outlook.com',
    description='The repository for the code completion project.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/miggymigz/code-completion',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'fire',
        'pygments',
        'requests',
        'tqdm',
    ]
)
