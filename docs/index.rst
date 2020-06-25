.. Chains documentation master file, created by
   sphinx-quickstart on Thu Jun 25 13:15:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Chains's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

	
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Chains object:
    A generic object to store data from MCMC chains. 
    It has convenience methods for savinig the chains to disk and to 
    calculate statistical properties of the chains.

    Based on the MultiNestResult object from StarKit
   
Methods:
	
from_array(cls, arr, parameter_names):
        
        Construct a Chains object from numpy array and list of parameters
        If there is no 'weights' value in the parameter_names, then assume
        the chains are equal weights. 

        Parameters
        ----------
        arr: numpy array (N points,n_parameters)

        Return
        ------
        Chains object
		
		
__init__(self, posterior_data):
        
        Create a Chains object from a pandas data frame. If there are no 'weights' column, then it 
        will create one with the same weights
		
		
from_hdf5(cls, h5_fname, key='chains'):
        
		Reading a result from its generated HDF5 file

        Parameters
        ----------

        h5_fname: ~str
            HDF5 filename

        key: ~str
            group identifier in the store
		
 
plot_triangle(self, parameters = None, sigma=None,debug=False,show_titles=True,
                      **kwargs):
        
        Produce a corner plot of the chains posterior.

        Keywords
        --------
        parameters - a list of paramters to plot. By default, it will plot all fit parameters. This is useful if you run into problems where one of the fit paramters is fixed and corner.py does not work on it
        
		sigma - limit the plot window to this many sigma from the median. This will help to zoom in on the region of interest for very large number of points.	
				
to_hdf(self, fname_or_buf, key='chains'):
        
		Writing the result out to HDF5.

        Parameters
        ----------

        fname_or_buf: ~str
            filename or buffer

        key: ~str
            key to save it under default='data'
		

1d_hist_plot(insert):
		
		Visualization of MCMC likelihood output.
	
	
save_output(insert):
		
		Stores MCMC output in text file.
