# ML-Project1

Machine Learning algorithm with own optimization functions to find the Higgs boson using original data from CERN. See the description of the project in Kaggle: https://www.kaggle.com/c/epfml-higgs.

* [Project Structure](#project-structure)
* [Project Requirements](#project-requirements)
* [How-To: Execute the Code](#how-to-execute-the-code)
    - [Executing the code from the Terminal](#executing-the-code-from-the-terminal)
    - [Executing the code from PyCharm](#executing-the-code-from-pycharm)
* [How-To: Include Functions of other files](#how-to-include-functions-of-other-files)
* [How-To: Comment Python Code](#how-to-comment-python-code)
* [How-To: Create Absolute Paths](#how-to-create-absolute-paths)

## Project Structure
The raw data must be keeped and unchanged (set permissions to only read for more security if you want) and it's too big to be uploaded to GitHub. The default directory is `../../data` and there will only be the original files downloaded from Kaggle.

## How-To: Execute the Code
#### Executing the code from the Terminal

Move to the root folder and execute:

    python train.py

The previous main call can also include several arguments that will condition the experiment and thus lead to different results. If you want to know which variables can be changed with the program call or what they do, execute one of the following two instructions:

    python train.py -h
    python train.py --help

#### Executing the code from PyCharm

We need to modify the default execution behavior. To do that:

1. Go to `Run -> Edit Configurations`
2. Click the + icon and select _"Python"_.
3. Change _"Script path"_ by _"Module name"_ and set it to _"hackmob"_.
4. Uncheck the _"Show command line afterwards"_ option (if checked).

If you want to execute a test module execute it as you always do. **If it doesn't work tell me**

## How-To: Include Functions of other files

To include a certain function from one specific file of the source `src` package we must use the following expression:

    from src.my_file include do_something
    
We can access modules from the `util` package as well:

    from utils.costs import compute_loss

## How-To: Comment Python Code

In order to create the documentation automatically, certain comment rules must be followed. The [Sphinx Domain](http://www.sphinx-doc.org/en/1.4.8/domains.html#basic-markup) is the standard way to comment Python Code. An example of a commented function is:

    def compute_sum(a,b):
        """
        Computes the sum of the input values
        
        :param a: first number
        :param b: second number
        :return: sum of the numbers
        """
        return a + b
A part of this, it's a good practice to comment lines with simples `# comment`especially for compact or hard-to-understand expressions.
