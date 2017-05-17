# Speech Recognition

Course material &amp; homeworks.

## Assignments

Each assignment has at least three files: setup.sh, requirements.txt and SR.py. 

First two of them are necessary for setting up environment and installing dependencies, the last one is for the assignment task itself. These steps below can be followed to use these files in order to build and run any assignment:

1. Clone the repository by typing `git clone https://github.com/emrekgn/speech-recognition.git` in the terminal.
2. Change directory to desired assignment directory by issuing `cd speech-recognition/assignment-X`
3. Execute `sudo ./setup.sh` script. This will install necessary packages.
4. Create virtual environment by issuing `virtualenv venv && source venv/bin/activate`. This will create isolated environment where all your dependencies are independent from the system-wide packages.
5. We use pip as Python package manager, type `pip install -r requirements.txt` in the terminal in order pip to install required Python dependencies.
6. Finally execute assignment script by typing `python SR.py` in the terminal. This will run assignment-related stuff and put any output file under output/ directory.

> After executing the assignment script, you can exit from the virtual environment by typing `deactivate` in the terminal.

Each assignment is executed and tested on Linux Mint 17 Qiana 64-bit!
