from numpy import arange,sqrt

def kaplan_meier(t,ttype,mean_only=False):
    """
      t: measurement array to be analyzed
      ttype: array, False or 0 if data point is right censored
                    True or 1 if data point measured
    """

