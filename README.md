# earthquakepy

A python library for earthquake engineers and seismologists.

## Installation

earthquakepy can be installed using pip. It’s an absolute breeze. Try it!

``` sh
pip install earthquakepy
```

Thats it! This will install the earthquakepy and other required libraries. Wasn’t that easy?


## Import and use

Probably you know how to import the library. Let me just remind you.

``` python
import earthquakepy as ep
```

## Getting started

+ Read a PEER NGA record from file

``` python
ts1 = ep.read_peer_nga_file(filename)
```

+ Read a raw file containing timeseries data

``` python
ts2 = ep.read_raw_file(filename)
```

+ Build a SDOF system object to carry out some magic later!

``` python
s1 = ep.sdof(T=1.0, xi=0.05)
```

+ Get response of above SDOF system subjected to base excitation ts2

``` python
s1.get_response(ts2)
```

+ Build a MDOF system object

``` python
import numpy as np

M = np.random.rand(3, 3)
C = np.random.rand(3, 3)
K = np.random.rand(3, 3)

m1 = ep.mdof(M=M, C=C, K=K)
```

+ Read OpenSees node output file

``` python
o1 = ep.read_ops_node_output(filename, 3, compNames=["x", "y", "z"])  # 3 : ncomps = number of components per node
```

+ Read OpenSees element output file

``` python
o2 = ep.read_ops_element_output(filename, 3, compNames=["x", "y", "z"])  # 3 : ncomps = number of components per element
```

+ Read OpenSees JSON model file

``` python
model = ep.read_opr_json_model(jsonFile)
```

Each object generated above has its own methods which are given in detail in the documentation. Please click on [this link](https://dbpatankar.github.io/earthquakepy) to view it.

