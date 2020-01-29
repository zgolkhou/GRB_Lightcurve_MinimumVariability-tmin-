from numpy import arange,sqrt

def kaplan_meier(t,ttype,mean_only=False):
    """
      t: measurement array to be analyzed
      ttype: array, False or 0 if data point is right censored
                    True or 1 if data point measured
    """
    n = len(t)
    ii  = t.argsort()

    # S = Prod ti<t 1-di/ni
    nn = arange(n,0,-1,dtype='float64')
    x = ttype[ii]/nn
    frac = (1-x).cumprod()

    # Greenwood's formula, V(S) = S^2 * Sum ti<=t di/ni/(ni-di)
    dfrac = frac * sqrt( (x/(1-x)/nn).cumsum() )
    if (ttype.sum()==n or x[-1]==1): dfrac[-1]=0

    if (mean_only):

        tt1 = ttype[ii]
        ii1 = ii[tt1]
        dt = t[ii1[1:]] - t[ii1[:-1]]
        dt0 = t[ii[-1]] - t[ii1[-1]]

        mn = ( dt * frac[tt1][:-1] ).sum() + dt0*frac[tt1][-1]
        fracl = frac-dfrac
        mnl = ( dt * fracl[tt1][:-1] ).sum() + dt0*fracl[tt1][-1]
        fracu = frac+dfrac
        mnu = ( dt * fracu[tt1][:-1] ).sum() + dt0*fracu[tt1][-1]

        t0 = t[ii1[0]]
        return mn+t0,mnl+t0,mnu+t0

    else:

        return t[ii], frac, dfrac
