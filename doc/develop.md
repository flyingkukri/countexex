# Developer Guide
This guide contains necessary information to extend or customize countexex to suit your specific needs.

## Overview
In this section we provide an overview of the software architecture of the tool.
## Mlpack
- Mlpack is column major in arma; Thus each column represents a data point
## Setup
In order to be able to debug the system set the option **STORM_DEVELOPER** to **ON** in **/countexex/storm/CMakeLists.txt**
## Extending countexex
### Supporting New Objectives
Currently only reachability objectives are supported ...