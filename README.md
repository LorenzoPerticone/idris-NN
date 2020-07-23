# idris-NN
Neural Networks in idris

This is a toy project as of now, but it should be enough to easily code a (*very* slow) feed-forward neural network.

It doesn't compile with idris2, but probably the only reason is it's missing the "HVect" library: I'll check if that's true in the future.

Things to do include:
 - cleanup (there *certainly* is some trash in the code)
 - improve documentation (the only documentation as of now consists of a few comments here and there)
 - improve memory and time efficiency (it's *very* slow and heavy)

The file "IdxReader.idr" should be a standalone library for parsing idx files, and will rely heavily on dependent tuples, but it's not yet complete.

The file "boston_reader.idr" is an example of usage (very simple, and it still doesn't divide the training set from the validation set).
