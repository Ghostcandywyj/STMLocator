# STMLocator
This is an original implementation of STMLocator of (Bug Localization via Supervised Topic Modeling)[...]

## Usage

python stmlocator -h
```
>  -inputB [INPUTB]    Input bug reports and their corresponding source files.
>  -inputS [INPUTS]    Input source files.
>  -output [OUTPUT]    Output file.
>  -alpha ALPHA        Super parameter for generating topics/source files.
>  -beta BETA          Super parameter for generating common words
>                      distribution.
>  -mu MU              Super parameter for generating co-occurrence words
>                      distribution.
>  -eta ETA            Super parameter for generating Bernoulli distribution
>                      \Psi.
>  -lenfunc [LENFUNC]  Length function of LOC. (Linear-[lin],
>                      Logarithmic-[log], Exponential-[exp], Square root-[srt])
>  -fold FOLD          Number of training folds.
>  -iter ITER          Number of training iterations for each fold.
```
