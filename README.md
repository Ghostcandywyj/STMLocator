# STMLocator
This is an original implementation of STMLocator of (Bug Localization via Supervised Topic Modeling)[...]

More information is able to added in the future.

Same copy can be found in our (group site)[https://github.com/SoftWiser-group/STMLocator]

## Usage
```
python stmlocator.py -h
```
```
  -inputB [INPUTB]    Input bug reports and their corresponding source files.
  -inputS [INPUTS]    Input source files and their LOC.
  -output [OUTPUT]    Output file.
  -alpha ALPHA        Super parameter for generating topics/source files.
  -beta BETA          Super parameter for generating common words
                      distribution.
  -mu MU              Super parameter for generating co-occurrence words
                      distribution.
  -eta ETA            Super parameter for generating Bernoulli distribution
                      \Psi.
  -lenfunc [LENFUNC]  Length function of LOC. (Linear-[lin],
                      Logarithmic-[log], Exponential-[exp], Square root-[srt])
  -fold FOLD          Number of training folds.
  -iter ITER          Number of training iterations for each fold.
```

## Sample
easy start with
```
python stmlocator.py -inputB pdereport.txt -inputS pdesource.txt -output pdeoutput.txt
```
