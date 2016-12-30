Modeling-Farm-Size-Distribution
==================================

Python files for a general equilibrium model of agricultural economy with market power distortions

by Jonathan Conning (building on work with Aleks Michuda)

Python code for variants of general equilibrium models of the farm size distribution.
Basic stuff ported from earlier Mathcad and Matlab files (some of it work with Aleks Michuda).

Variations on a neo-classical model to explore how factor endowments, the initial distribution of property rights
and skills in the population interact with production technologies to shape equilibrium patterns of agrarian
production organization and the size distribution of farms. To understand the main arguments, consider the
simplest case of a single landlord (or a cartel of landlords) surrounded by a fringe of small landowning or
landless agricultural households. If the lanlord owns a large fraction of the land endowment a standard
partial-equilibrium analysis of non-price discriminating monopoly suggests the landlord would drive up
the rental price of land by withholding land from the lease market. In a general equilibrium setting however there
is another effect: by restricting other farmers' access to land landlords also lower the marginal product of labor
on those farms. This increases the supply of labor to landlord estates at any given wage increasing landlords'
potential income from monopsony rents. This can lead to equilibria where landlords increase the size of their
production estates scale well above efficient scale in a competitive economy. A Latifundia-Minifundia type
economy can emerge in which landlords operate large estates employing overly land-intensive production techniques
while a large mass of farmers operate inefficiently and labor-intensive small parcels of land and sell
labor to the landlord estate(s).

The model in python builds off similar earlier efforts in Mathcad and MATLAB. Details in the
appendix at the end of this notebook.

I use object oriented programming ideas, first defining a "class" of Economy. An instance of an economy
is an object with atributes including an associated endowment and technology as well as an initial distribution
of property rights and non-traded skills. The economy class includes methods for finding a vector of
market-clearing factor prices and associated equilibrium net factor demands and outputs.

I will later define a subclass PowerEconomy which inherits all the attributes of the Economy class but
adds a few methods to compute market-power distorted equilibria.
