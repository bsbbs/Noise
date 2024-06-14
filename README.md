# Origins of noise in both improving and degrading decision making

[![arXiv shield](https://img.shields.io/badge/arXiv-1709.01233-red.svg?style=flat)]([https://doi.org/10.1101/2024.03.26.586597])
![Static Badge](https://img.shields.io/badge/bioRXiv-10.1101%2F2024.03.26.586597-red.svg?style=flat&link=https%3A%2F%2Fdoi.org%2F10.1101%2F2024.03.26.586597)


__Bo Shen, Jailyn Wilson, Duc Nguyen, Paul Glimcher, Kenway Louie__

__New York University, Grossman School of Medicine__

---

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [License](./LICENSE)
- [Citation](#citation)

# Overview

This is a repository for a manuscript under review, titled "Origins of noise in both improving and degrading decision making". This is an unpublished work. The contents may change during the revision.

# Repo Contents

- [Empirical data](./myData): Human choice data created by the current project. Please check the [README file](./myData/README.txt) for further information about the data structure and variables.
- [Simulation code](./ModelSimulationCode): Matlab code for the modeling part of the paper. To replicate the simulation, please follow the main files [Fig1](./ModelSimulationCode/Fig1.m), [Fig2](./ModelSimulationCode/Fig2.m), and [Fig4](./ModelSimulationCode/Fig4.m).
- [Data analysis code](./BehavioralDataAnalysisCode.Rmd): `R` code for analyzing the empirical data. To replicate the data analysis, please change the directory of the data according to your local directory environment and follow the sections in the code.

# System Requirements

## Hardware Requirements

The [Matlab code](./)`lol` package requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 2 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
CPU: 4+ cores, 3.3+ GHz/core

The runtimes below are generated using a computer with the recommended specs (16 GB RAM, 4 cores@3.3 GHz) and internet of speed 25 Mbps.

## Software Requirements

### OS Requirements

The package development version is tested on *Linux* and  operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 16.04  
Mac OSX:  
Windows:  

The CRAN package should be compatible with Windows, Mac, and Linux operating systems.

# Citation

For usage of the package and associated manuscript, please cite our [bioRXiv preprint] (https://www.biorxiv.org/content/10.1101/2024.03.26.586597v2).


