# -*- coding: utf-8 -*-
""" geqfarm.py   General Equilibrium Farm Size Distribution
An Economy Class and methods for calculating and representing General
equilibrium models of the farm size distribution with and without factor
market distortions.

Authors: Jonathan Conning & Aleks Michuda
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from collections import namedtuple


class Economy(object):

    """ Economy with an Equilibrium Farm Size Distribution
    Args:
         N (int): number of farm-size classes or bins in the distribution

    Note: We take the landlord class to be last [-1] indexed group.
    By default the initial distribution of skills is uniformly distributed.
    For example N = 5 and s = np.array([1, 1, 1, 1, 1.5]) has 5 farmer groups.
    But any distribution can be used.
    """

    def __init__(self, N):  # constructor to set initial default parameters.
        self.N       = N   # of xtiles (number of skill groups)
        self.GAMMA   = 0.8    # homogeneity factor
        self.ALPHA   = 0.5    # alpha (land) for production function
        self.LAMBDA  = 1.0/N  # landlord share of labor
        self.TBAR    = 100    # Total Land Endowment
        self.LBAR    = 100    # Total Labor Endowment
        self.H       = 0.0    # fixed cost of production
        self.s       = np.ones(N)

    def __repr__(self):
        return 'Economy(N={}, GAM={}, TBAR={}, LBAR={})'.format(self.N, self.GAMMA, self.TBAR, self.LBAR)

    def prodn(self, X, s):
        """ Production function
        Args:
            X:  vector of factor inputs (X[0] land and X[1] labor)
            s:  vector of skill endowments by xtile
        Returns:  vector of output(s)
        """
        Y = s*((X[0]**self.ALPHA)*(X[1]**(1-self.ALPHA)))**self.GAMMA
        return Y

    def marginal_product(self, X, s):
        """ Production function technoogy
        Args:
            X:  vector of factor inputs (X[0] land and X[1] labor)
            s:  vector of skill endowments by xtile
        Returns:  vector of marginal products
        """
        MPT = self.ALPHA*self.GAMMA*self.prodn(X,  s)/X[0]
        MPL = (1-self.ALPHA)*self.GAMMA*self.prodn(X, s)/X[1]
        return np.append(MPT, MPL)

    def profits(self, X, s, w):
        """ profits given factor prices and (T, L, s)"""
        return self.prodn(X, s) - np.dot(w, X) - self.H

    def demands(self, w, s):
        """Competitive factor demands for each skill group in a subeconomy
        Args:
            w:  vector of factor prices (w[0] land rent and w[1] wage)
            s:  vector of skill endowments by xtile
        Note:  Farms with negative profits assumed to shut down with zero demands.
        Returns:  vector of factor demands
        """
        alpha, gamma = self.ALPHA, self.GAMMA
        land = ((w[1]/(gamma*s*(1-alpha))) *
                (((1-alpha)/alpha)*(w[0]/w[1])) **
                (1-gamma*(1-alpha)))**(1/(gamma-1))
        labor = ((w[0]/(gamma*s*alpha)) *
                 ((alpha/(1-alpha))*(w[1]/w[0])) **
                 (1-gamma*alpha))**(1/(gamma-1))
        # if fixed cost implies negative profits, zero demands
        X = np.array([land, labor])
        profitable = (self.profits(X, s, w) > 0)
        return X*profitable

    def excessD(self, w, Xbar, s):
        """ Total excess land and labor demand given factor prices in
        subeconomy with Xbar supplies
        returns excess demand in each market
        """
        res = np.array(([np.sum(self.demands(w, s)[0])-Xbar[0],
                         np.sum(self.demands(w, s)[1])-Xbar[1]]))
        return res

    def smallhold_eq(self, Xbar, s, analytic=True):
        """ Solves for market clearing factor prices in economy with Xbar supplies.

        Solve analytically or numerically (minimizes sum of squared excess demands)
        Args:
            X:  vector of factor inputs (X[0] land and X[1] labor)
            s:  vector of skill endowments by xtile
            analytic (bool): by default solve analytically
        Returns:
            res (named tuple): factor prices and demands res.w, res.X
        """

        if analytic:     # for specific CobbDouglas
            gamma = self.GAMMA
            s_fringe, s_R = s[0:-1], s[-1]
            psi = np.sum((s_fringe/s_R)**(1/(1-gamma)))
            Lr = Xbar[1]/(1+psi)
            Tr = Xbar[0]/(1+psi)
            L_fringe = Lr*(s_fringe/s_R)**(1/(1-gamma))
            T_fringe = Tr*(s_fringe/s_R)**(1/(1-gamma))
            Xs = np.array([np.append(T_fringe, Tr), np.append(L_fringe, Lr)])
            WR = self.marginal_product(Xs[:, -1], s[-1])
        else:  # Numeric solution should work for any demands
            w0 = np.array([0.5, 0.5])

            def f(w):
                return np.sum(self.excessD(w, Xbar, s)**2)

            res = minimize(f, w0, method='Nelder-Mead')
            WR = res.x
        Xs = self.demands(WR, s)
        result = namedtuple('result', ['w', 'X'])
        res = result(w=WR, X=Xs)
        return res

    def cartel_income(self, Xr, theta):
        """ Cartel group's income from profits and factor income

        when cartel uses (tr,lr) fringe has (TBAR-tr,LBAR-lr)  """
        # at present cartel is always last index farm
        s_fringe, s_R = self.s[0:-1], self.s[-1]  # landlord is last farmer
        TB_fringe = max(self.TBAR - Xr[0], 0)
        LB_fringe = max(self.LBAR - Xr[1], 0)
        fringe = self.smallhold_eq([TB_fringe, LB_fringe], s_fringe)
        y = self.prodn(Xr, s_R) - \
            np.dot(fringe.w, [Xr[0]-self.TBAR*theta,
                              Xr[1]-self.LAMBDA*self.LBAR])
        # print("cartel:  Tr={0:8.3f}, Lr={1:8.3f}, y={2:8.3f}".format(Xr[0],
        # Xr[1],y))
        return y

    def cartel_eq(self, theta, guess=[1, 1]):
        """ Cartel chooses own factor use (and by extension how much to
        withold from the fring to max profits plus net factor sales)
        """
        def f(X):
            return -self.cartel_income(X, theta)
        res = minimize(f, guess, method='Nelder-Mead')
        XR = res.x
        # print('XR:',XR)
        fringe = self.smallhold_eq([self.TBAR, self.LBAR]-XR, self.s[0:-1])
        XD = np.vstack((fringe.X.T, XR)).T
        WR = fringe.w

        result = namedtuple('result', ['w', 'X'])
        cartel_res = result(w= WR, X= XD)
        return cartel_res

    def print_params(self):
        """ print out parameters alphabetically"""
        params = sorted(vars(self).items())
        for itm in params:
            print(itm[0], '=', itm[1], end=', ')


class CESEconomy(Economy):
    """ sub class of Economy class but with two factor CES
    """
    def __init__(self, N):  # constructor to set initial parameters.
        super(CESEconomy, self).__init__(N)  # inherit properties
        # if None supplied use defaults
        self.N         = N # of quantiles (number of skill groups)
        self.RHO       = 0.8    # homogeneity factor
        self.PHI       = 0.5    # alpha (land) for production function
        self.aL        = 1.0    # landlord share of labor
        self.aT        = 1.1    # Total Land Endowment

    def __repr__(self):
        return 'CESEconomy(N={}, GAM={}, TBAR={}, LBAR={})'.format(self.N, self.GAMMA, self.TBAR, self.LBAR)

    def prodn(self, X, s):
        Y = s*(self.PHI*X[0]**(self.RHO) + (1-self.PHI)*X[1]**(self.RHO))  \
            ** (self.GAMMA/self.RHO)
        return Y

    def marginal_product(self, X, s):
        """ Production function technoogy """
        common = s*(self.PHI*X[0]**self.RHO+(1-self.PHI)*X[1]**self.RHO) \
            ** ((1+self.RHO)/self.RHO)
        MPT = common * self.PHI*X[0]**(-self.RHO-1)
        MPL = common * (1-self.PHI)*X[1]**(-self.RHO-1)
        return np.append(MPT, MPL)

# End of class definitions

def scene_print(ECO, numS=5,prnt=True,detail=True):
        """Creates numS land ownership scenarios by varying land gini THETA
        calculating competitive and market-power distorted equilibria for each
        Prints results if flags are on.

        Args:
          ECO -- Instance of an Economy object
          numS -- number of values of theta
          prnt -- print table if True
        Returns:
          [Xc,Xr,wc,wr]  where
            Xc -- Efficient/Competitive landlord factor use
            Xr -- numS x 2 matrix, Xr[theta] = Landlords' distorted use
            wc -- competitive factor prices
            wr -- wr[theta] distorted competitive factor prices

        """
        print(("Running {0} scenarios...".format(numS)))
        # competitive eqn when landlord is just part of the competitive fringe
        comp = ECO.smallhold_eq([ECO.TBAR,ECO.LBAR],ECO.s)
        wc, Xc = comp.w, comp.X
        Xrc = Xc[:,-1]   # landlord's factor use

        #
        guess = Xrc
        # distorted equilibria at different land ownership theta
        theta = np.linspace(0,1,numS+1)
        theta[-1] = 0.97
        if prnt:
            print("\nAssumed Parameters")
            print("==================")
            ECO.print_params()
            print(('\nEffcient:[ Trc, Lrc]      [rc,wc]       w/r   '), end=' ')
            if detail:
                print(('F( )    [r*Tr]  [w*Lr]'), end=' ')
            print("")
            print(("="*78))
            print(("        [{0:6.2f},{1:6.2f}] ".format(Xrc[0],Xrc[1])), end=' ')
            print(("[{0:4.2f},{1:4.2f}]".format(wc[0], wc[1])), end=' ')
            print(("  {0:4.2f} ".format(wc[1]/wc[0])), end=' ')
            if detail:
                print(("| {0:5.2f} ".format(ECO.prodn(Xrc,ECO.s[-1]))), end=' ')
                print((" {0:5.2f} ".format(Xrc[0]*wc[0])), end=' ')
                print((" {0:6.2f} ".format(Xrc[1]*wc[1])))

            print(("\nTheta  [ Tr, Lr ]      [rM,wM]        w/r  |"), end=' ')
            print('F()   [T_hire]  [T_sale] [L_hire]')

            print(("="*78))

        Xr = np.zeros(shape=(numS+1, 2))  # Xr - load factor use for each theta
        Tr, Lr = np.zeros(numS + 1), np.zeros(numS + 1)  # Xr - load factor use for each theta
        rw = np.zeros(shape=(numS+1,2))
        w, r = np.zeros(numS + 1), np.zeros(numS + 1)

        for i in range(numS+1):
            cartelEQ = ECO.cartel_eq(theta[i], guess)
            Xr[i] = cartelEQ.X[:, -1]
            Tr[i], Lr[i] = Xr[i]
            rw[i] = cartelEQ.w
            r[i], w[i] = rw[i]
            guess = Xr[i]
            if prnt:
                print((" {0:3.2f}".format(theta[i])), end=' ')
                print((" [{0:6.2f},{1:6.2f}]".format(Tr[i],Lr[i])), end=' ')
                print(("[{0:5.2g},{1:5.2f}] {2:5.2f}" \
                .format(r[i],w[i],w[i]/r[i])), end=' ')
                if detail:
                    print(("| {0:5.2f} ".format(ECO.prodn(Xr[i],ECO.s[-1]))), end=' ')
                    print((" {0:6.2f} ".format(Xr[i,0]*rw[i,0])), end=' ')
                    print((" {0:6.2f} ".format(theta[i]*ECO.TBAR*rw[i,0])), end=' ')
                    print((" {0:6.2f} ".format(Xr[i,1]*rw[i,1])), end=' ')
                print("")
        if prnt:
            print(("="*78))

        return (Xrc, Xr, wc, rw)


def factor_plot(ECO, Xrc, Xr):
    plt.rcParams["figure.figsize"] = (10, 8)
    numS = len(Xr)-1
    theta = np.linspace(0, 1, numS+1)
    Tr, Lr = Xr[:, 0], Xr[:, 1]
    Tr_net = Tr-np.array(theta) * ECO.TBAR
    Lr_net = Lr - ECO.LAMBDA * ECO.LBAR
    # print(Tr_net, Lr_net)
    Trc_net = Xrc[0]*np.ones(numS+1)-np.array(theta)*ECO.TBAR
    Lrc_net = Xrc[1]*np.ones(numS+1)-ECO.LAMBDA*ECO.LBAR
    plt.grid()
    plt.plot(theta, Tr_net, '-ro', label='distorted land')
    plt.plot(theta, Trc_net, label='efficient land')
    plt.plot(theta, Lr_net, '-b*', label='distorted labor')
    plt.plot(theta, Lrc_net, label='efficient labor')
    plt.grid()
    plt.ylim(-100, ECO.TBAR)
    # plt.xlabel(r'$\gamma =$')
    plt.title('Landlord net factor hire for '+r'$\gamma =$ {0}'
              .format(ECO.GAMMA))
    plt.xlabel(r'$\theta$ -- Landlord land ownership share')
    plt.legend(loc='lower left',title='net hiring in of')
    plt.show()
    return

def TLratio_plot(ECO, Xrc, Xr):
    plt.rcParams["figure.figsize"] = (10, 8)
    numS = len(Xr)-1
    theta = np.linspace(0, 1, numS+1)
    plt.plot(theta, Xr.T[0][:]/Xr.T[1][:], '-ro', label='distorted')
    plt.plot(theta, (Xrc[0]/Xrc[1])*np.ones(numS+1), '--', label='efficient')
    plt.title('Land to labor ratio on landlord farm '+r'$\gamma =$ {0}'
              .format(ECO.GAMMA))
    plt.xlabel(r'$\theta$ -- Landlord land ownership share')
    plt.legend(loc='upper left',title='Land/Labor ratio')
    plt.show()
    return

#
if __name__ == "__main__":
    """Sample use of the Economy class """

    s = np.array([1.,  1.,  1.,  1.,  1.])
    N = len(s)
    E = Economy(N)    # an instance takes N length as parameter
    E.ALPHA = 0.5
    E.GAMMA = 0.90

    E.smallhold_eq([E.TBAR, E.LBAR], s, analytic=True)

    (Xrc, Xr, wc, wr) = scene_print(E, 10, detail=True)

    factor_plot(E,Xrc,Xr)
    TLratio_plot(E,Xrc,Xr)
