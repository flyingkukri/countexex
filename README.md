# Counterexample explanation in MDPs using Storm
lorem ipsum
## Building
### Dependencies
- Storm (https://www.stormchecker.org/documentation/obtain-storm/build.html)
- mklearn (on fedora mklearn-devel)
### Compiling
    mkdir build
    cd build
    cmake ..
    make
## Running
Supply a model at the end of which the label goal is given to the states F, e.g.
    label "goal" = s=1|s=2;
Run the following command:
    countexex model.nm