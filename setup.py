# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = "0.1.0.dev"

INSTALL_REQUIRES = [
    "transformers>=4.26.0",
    "torch",
    "scipy",
    "numpy",
]


setup(
    name="positional_hidden",
    version=VERSION,
    author="The Positional Hidden team",
    author_email="yuyj22@mails.tsinghua.edu.cn",
    description="To mitigate position bias in LLMs, especially in long-context scenarios, we scale only one dimension of LLMs, reducing position bias and improving average performance by up to 15.4%.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    keywords="LLMs, Long-context LLMs, Positional bias",
    license="MIT License",
    url="https://github.com/PositionalHidden/PositionalHidden",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
)
