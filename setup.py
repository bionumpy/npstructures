#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages # , Extension

#module = Extension('npstructures.copy_segment', sources=['npstructures/copy_segment.pyx'])
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.19']

test_requirements = ['pytest>=4.6', 'hypothesis']

setup(
    author="Knut Rand",
    author_email='knutdrand@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Simple data structures that augments the numpy library",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='npstructures',
    name='npstructures',
    packages=find_packages(include=['npstructures', 'npstructures.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/knutdrand/npstructures',
    version='0.2.16',
    zip_safe=False,
    # ext_modules=[module],
)

# python -m build
# twine upload dist/*
