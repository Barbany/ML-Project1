# ML-Project1

Machine Learning algorithm with own optimization functions to find the Higgs boson using original data from CERN. See the description of the project in Kaggle: https://www.kaggle.com/c/epfml-higgs.

* [Project Structure](#project-structure)
* [Project Requirements](#project-requirements)
* [How-To: Execute the Code](#how-to-execute-the-code)
    - [Executing the code from the Terminal](#executing-the-code-from-the-terminal)
* [How-To: Include Functions of other files](#how-to-include-functions-of-other-files)
* [How-To: Comment Python Code](#how-to-comment-python-code)
* [How-To: Create Absolute Paths](#how-to-create-absolute-paths)

## Project Structure
#### Data
The raw data must be keeped and unchanged (set permissions to only read for more security if you want) and it's too big to be uploaded to GitHub. The default directory is `../data` and there will only be the original files downloaded from Kaggle.

We may implement preprocessing and cleaning techniques such as substituting meaningul values (indicated with -999.0 in this dataset) with NA. Even with this simple action, create a script or python file and save it to recreat it a near future if needed. Remember not to change the original data so save the file in another folder.

It may also be convinient to have a reduced dataset of a few instances (either processed or not) to test our algorithm without spending much time every time we do a change that could prompt potential errors.

#### Dependencies
For the first time you create your environments, install all dependencies with the file `requirements.txt` and every time you need a new library add it to the file.

Every time you think of an improvement or a modification, please start an issue even if you are starting to work with it. This will show the issue in the project management tab and others will be able to contribute.

#### Commits and branches
Every commit should have a title and if necessary, an explanatory description. Each commit should not be very long so I recommend not to do one that changes more than for example 20 lines (we may want to undo it in a future if this don't work). Once the project works and the first results come, you should create a branch for doing experiments to avoid corrupting a functional algorithms. If you achieve better results than the original code, a merge could happen.

#### Documentation
The root of the project have a `doc/`directory to store all the documentation needed for the project. If you check any paper, put the references in there as well as the `.bib` citation file for the final report.

#### Report
If you want to use the online LaTex editors OverLeaf, you can directly import the `.tex` file from GitHub by linking this repository. Note that all changes won't be commited if you don't commit by going to `Menu>Sync>Github`. In this case, only the `.tex` file is changed in the repository, so don't try to find an updated PDF.

#### Naming variables
Give functions and variables meaningful names both to document their purpose and to make the program easier to read. While it’s acceptable to call the counter variable in a loop i or j, the major data structures in a program should not have one-letter names. Remember to follow each language’s conventions for names, such as `net_charge` for Python.

## How-To: Execute the Code
#### Executing the code from the Terminal

Move to the root folder and execute:

    python runn.py

The previous main call can also include several arguments that will condition the experiment and thus lead to different results. If you want to know which variables can be changed with the program call or what they do, execute one of the following two instructions:

    python run.py -h
    python run.py --help

## How-To: Include Functions of other files

To include a certain function from one specific file of the source `src` package we must use the following expression:

    from src.my_file include do_something
    
We can access modules from the `util` package as well:

    from utils.costs import compute_loss

## How-To: Comment Python Code

Place a brief explanatory comment at the start of every program. In order to create the documentation automatically, certain comment rules must be followed. The [Sphinx Domain](http://www.sphinx-doc.org/en/1.4.8/domains.html#basic-markup) is the standard way to comment Python Code. An example of a commented function is:

    def compute_sum(a,b):
        """
        Computes the sum of the input values
        
        :param a: first number
        :param b: second number
        :return: sum of the numbers
        """
        return a + b
A part of this, it's a good practice to comment lines with simples `# comment`especially for compact or hard-to-understand expressions.
