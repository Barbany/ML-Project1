# ML-Project1

Machine Learning algorithm without external libraries to find the Higgs boson using original data from CERN (Kaggle challenge). First project for Machine Learning course (CS-443) at EPFL.

* [Getting started](#getting-started)
* [Running the code](#running-the-code)

## Getting started
#### Data
The raw data can be downloaded form the webpage of the Kaggle challenge: https://www.kaggle.com/c/epfml18-higgs. The default directory to locate the uncompressed .csv files is `data/`, which is an empty folder by now. Nevertheless, you can change this location by modifying the default parameters located in the file `src/utils/argument_parser.py` and use your own path.

The processed data as well as a log of all prints will be located in a folder named `results/experiment_name/`, where the `experiment_name` keyword will be substituted by a string that indicates the parameters of the experiment that affect data processing. This means that the original files downloaded from Kaggle must remain unchanged.

#### Documentation
The root of the project have a `doc/` directory where you can find the guidelines of the project given by EPFL professors as well as a the explanation of the dataset. Check this to know more about the features and they relations. Note that this file has some of the lines of appendix B higlighted to justify one of the key improvements of the dataset to handle meaning-less data.

#### Dependencies
For the first time you can create your environment by installing all dependencies listed in the file `requirements.txt`. Note that the needed libraries are very basic because the machine learning methods used in this project are implemented in this same repository. You will therefore probably have them installed but make sure to satisfy the version requirement of numpy because we use the function quartiles, which was not included in early releases.

#### Report
Check our paper located at the `report/` folder to have an overview of the project and the justification of all the steps that leaded us to a model with the final accuracy presented in the Kaggle competition.

## Running the code
Move to the root folder and execute:

    python run.py

The parameters that define the model used for the last Kaggle submission are set by default. Nonetheless, we recommend to run it with verbose to see the evolution of the loss and to check the results for every data chunk:
    
    python run.py -verbose

The main call can also include several arguments that will condition the experiment and thus lead to different results. If you want to know which variables can be changed and what they do, execute one of the following two instructions:

    python run.py -h
    python run.py --help
