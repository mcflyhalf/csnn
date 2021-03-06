# CSNN

Implement image classifier for MNIST dataset without using any Neural Network library.

## Getting Started

To start, on your terminal, cd to the place where you want the project to live then:
```
$: sudo apt-get install python3
$: git clone https://github.com/mcflyhalf/csnn
$: virtualenv --PYTHON=python3 csnn
$: cd csnn
$: source bin/activate
$: pip3 install numpy
$: cd csnn
$: wget http://deeplearning.net/data/mnist/mnist.pkl.gz
$: cd ..
$: pip3 install -e .
```
Don't forget the period(.) at the end of the last command

In case you are curious, what you just did was to:
1. Download the git repository and files
1. Create a virtual environment to run python3
1. Activated your virtual environment (steps 3 and 4)
1. Installed numpy (whihc is the only 3rd party library we will use)
1. Downloaded the mnist dataset into the same directory as the files (if this doesnt work, go [here](http://deeplearning.net/data/mnist) and save manually to csnn/csnn)

Now open csnn/csnn and open all the python files in there in your favourite editor. Do not close your terminal.

At this point, your directory structure should be
