# Counterexample explanation in MDPs using Storm
lorem ipsum
## Building
### Dependencies
- Storm (https://www.stormchecker.org/documentation/obtain-storm/build.html)
- mklearn (on fedora mklearn-devel)
### Cloning and Compiling
    # because we have submodules 
    git clone git@github.com:flyingkukri/stormdt.git --recursive
    mkdir build
    cd build
    cmake ..
    make
## Running
Run the following command:
    countexex model.nm
### Model requirements
- Supply a model at the end of which the label goal is given to the states F, e.g.
    label "goal" = s=1|s=2;
- All final states are sink states

## Development
[Click here for information regarding our code structure](doc/develop.md)