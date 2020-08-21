# DomWorld: network motifs and dominance hierachies
Final project for the Advanced Self-organization of Social Systems course, Faculty of Science and Engineering, RUG University. In this project we have implemented some popular network motifs analysis techniques to investigate the effect that dominance hierarchies could have on the dominance network structure (in terms of triadic patterns), in a group of individuals.

## Parameters
The experiment has been performed with different values for the following domWorld parameters (`DomWorld Legacy model v2.0` version has been used):
- **group size:** from 8 to 48 individuals (at an equal sex ratio);
- **fleeing distance:** from 2.0 (default domWorld value), to 10.0 units;
- **intensity of aggression:** mild (0.1 females 0.2 males) and fierce (0.8 females 1.0 males) species values, as in Hemelrijk (1999).

## Installing
The code has been written with `python 3.7.7`, that can be downloaded [here](https://www.python.org/downloads/release/python-377/). The additional packages needed are specified in the *requirements.txt* file, and can be installed with the following terminal command:

```shell
python3 -m pip install -r requirements.txt

# if only python3 is installed
python -m pip install -r requirements.txt
```
