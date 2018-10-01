# ML-Project1

Machine Learning algorithm with own optimization functions to find the Higgs boson using original data from CERN.

* [Project Structure](#project-structure)
* [Project Requirements](#project-requirements)
* [How-To: Execute the Code](#how-to-execute-the-code)
    - [Executing the code from the Terminal](#executing-the-code-from-the-terminal)
    - [Executing the code from PyCharm](#executing-the-code-from-pycharm)
* [How-To: Include HackMob Modules](#how-to-include-hackmob-modules)
* [How-To: Comment Python Code](#how-to-comment-python-code)
* [How-To: Create Absolute Paths](#how-to-create-absolute-paths)

## Project Structure

#### Executing the code from the Terminal

Move to the root folder and execute:

    python3 hackmob

#### Executing the code from PyCharm

We need to modify the default execution behavior. To do that:

1. Go to `Run -> Edit Configurations`
2. Click the + icon and select _"Python"_.
3. Change _"Script path"_ by _"Module name"_ and set it to _"hackmob"_.
4. Uncheck the _"Show command line afterwards"_ option (if checked).

If you want to execute a test module execute it as you always do. **If it doesn't work tell me**

## How-To: Include HackMob Modules

To include modules of the `hackmob` package we must use the following notation:

    from hackmob.my_module include MyClass
    
We can access modules from the `hackmob_tests` package as well:

    from hackmob_tests.my_test_module include MyClass
    
Remember that the project folder must be in the **$PYTHONPATH** environment variable.

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
