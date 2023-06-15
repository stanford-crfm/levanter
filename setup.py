# get path to current directory
import os

import setuptools


curdir = os.path.dirname(os.path.abspath(__file__))

with open(f"{curdir}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_data_files():
    data_files = []
    for dirpath, dirnames, filenames in os.walk("config"):
        files = [os.path.join(dirpath, filename) for filename in filenames]
        data_files.append((dirpath, files))
    return data_files


# raise Exception(f"{get_data_files()}")

setuptools.setup(
    name="levanter",
    version="0.0.1",
    description="Jax-based training of foundation models",
    url="https://github.com/stanford-crfm/levanter",
    author="David Hall",
    author_email="dlwh@cs.stanford.edu",
    # install_requires=install_reqs,
    long_description=long_description,
    packages=setuptools.find_packages(where="src", exclude=("tests",)),
    # https://stackoverflow.com/questions/70777486/pip-install-e-doesnt-allow-to-import-via-package-dir
    # package_dir={"": "src/", "levanter.configz": "config/"},
    package_dir={"": "src/"},  # , "levanter.config": "config/"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS X",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    # data_files=[("config", ["config/gpt2_small.yaml"])],
    # package_data={"levanter.config": ["*.yaml"]},
    # data_files=get_data_files(),
)
