# Homework 4 - Hard coding
## Authors
* Michele Conti
* Alessio Galimi
* Adrienn Timea Aszalos

*for "Algorithmic Methods of Data Mining", "La Sapienza" University of Rome, MSc in Data Science, 2020-2021*.
## Task
The goal of this homework was to implement important algorithms from sctach. We were challenged with two completely and interesting problems: the first exercise asks us to build an HyperLogLog structure (starting from the implementation of an hash function itself), while the second challenges us with the implementation of the K-Means clustering algorithm. Alongside the main tasks, we are also asked to answer several questions on the behaviour of these algorithms and on the analysis of the results we obtain using them.
## Usage
In the repository, it is included `requirements.txt`, which consists in a file containing the list of items to be installed using conda, like so:

`conda install --file requirements.txt`

Once the requirements are installed, you shouldn't have any problem when executing the scripts. Consider also creating a new environment, so that you don't have to worry about what is really needed and what not after you're done with this project. With conda, that's easily done with the following command:

`conda create --name <env> --file requirements.txt`

where you have to replace `<env>` with the name you want to give to the new environment.

Notice that, unfortunately, this kind of requirements file is built on a Linux machine, and therefore it is not guaranteed that this will work on different S.O.

## Repo structure
The repository consists of the following files:
* [__`main.ipynb`__](../main/main.ipynb):
    > This is the core of this repository. In fact it contains the results of our implementations and researches. Notice that this notebook contains just the answers to the questions asked in the homework assignment, not the actual code for it.
* [__`functions`__](../main/functions):
    > This directory contains the implementation of the functions we call in the main Jupyter Notebook file.
* [__`requirements.txt`__](../main/requirements.txt):
    > A txt file containing the dependecies of the project; see the usage part for details.