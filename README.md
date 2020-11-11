# DomWorld: network motifs and dominance hierachies
Final project for the Advanced Self-organization of Social Systems course, Faculty of Science and Engineering, RUG University.

## Parameters
The experiment has been performed with different values for the following domWorld parameters (`DomWorld Legacy model v2.0` has been used):
- **group size:** from 8 to 48 individuals (at an equal sex ratio);
- **fleeing distance:** from 2.0 (default domWorld value), to 10.0 units;
- **intensity of aggression:** mild (0.1 females 0.2 males) and fierce (0.8 females 1.0 males) species values, as in Hemelrijk (1999).

## Installing
The code has been written with `python 3.7.7`, that can be downloaded [here](https://www.python.org/downloads/release/python-377/). The additional packages needed are specified in the *requirements.txt* file, and can be installed with the following terminal command:
- if working on linux you may need [Wine]() to run `DomWorld Legacy model`, which was only available as a windows *.exe* file when I wrote this code.

```shell
# on linux
python3 -m pip install -r requirements.txt

# on windows
python -m pip install -r requirements.txt
```
