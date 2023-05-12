# ML Project

This code was made for a final project in the COM S 574 Machine Learning Course at Iowa State University. The main purpose of the code is to extract a number of features for TESS light curve files to classifiy their pulsations. Since this was a class project it is in a very 'alpha' state with poor commenting, readability, and description. 

The features and classes pickled in 'featured.npy' and 'classes.npy' can be fed almost directly into the sci-kit learn decision tree implimentation. Extracting the features takes a considerable amount of time since TESScut is queried for each object and multiple lomb-scargle instances are run to get powers, frequencies, and amplitudes.  
