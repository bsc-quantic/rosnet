### Deployment

We can deploy the rosnet's desktop version either in Windows, Linux or MacOS. A Python virtual environment is hardly recommended to deploy rosnet. To do so, we will need first to have a [Python installation](https://www.python.org/downloads/) in our computer. The minimum version is Python 3.8.2.

## Virtual environment

We will open a terminal in our working directory and do:

```bash
$ git clone https://github.com/UB-Quantic/rosnet.git
$ cd rosnet
```

To create a virtual environment and activate it we will do:

```bash
$ python3 -m venv env
$ source env/Scripts/activate
```

An `env` prefix should appear in our command line. Now we are using the virtual environment that have just created. If we make `$ which python`, the answer should be `{workingDirectory}/env/Scripts/python`. For a complete tutorial about virtual environments see [this link](https://realpython.com/python-virtual-environments-a-primer/).

Whenever you want to return to use the default Python and libraries in your computer, you should just run `$ deactivate`. For now, we will stay in our virtual environment, though.

## Needed libraries

We will need to install the following basic libraries within our virtual environement: numpy, multimethod, autoray.

```bash
$ python -m pip install --upgrade pip
$ pip3 install numpy multimethod autoray
```

Then we will be able to import them and start checking out rosnet right away. See a [first example](first_example.md).