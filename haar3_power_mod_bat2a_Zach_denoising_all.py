#! /usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Thu Aug. 27 00:48:00 2012

@author: Zach
"""
#import sys
#sys.modules[__name__].__dict__.clear()
#import Image
# import numpy as np
from numpy import *
#from numpy import log,loadtxt,arange,asarray,unique,concatenate,zeros,sqrt,ceil
import haar_nondec
import rate_rebin
import do_rebin
import haar_denoise            # Calling the haar_denoising modules
import matplotlib.pyplot as plt
import mu0_minimize_CHI2_fmin
# import scipy as sp
# import math
#file_input = "00104298_bat_fine_wlc.txt"

def haar2_power_mod2a_Zach_denoising(file_input): 

    def shift(l, n):
        return l[n:] + l[:n]
         
    text_file1 = open("out2put_tMIN_10_23.txt", "a")
    text_file2 = open("out2put_chi2_10_23.txt", "a")
    text_file3 = open("out2put_name_10_23.txt", "a")
    text_file4 = open("out2put_tminMIN_10_23.txt", "a")
    text_file5 = open("out2put_tmin_array_10_23.txt", "a")
    text_file6 = open("out2put_cvSize_10_23.txt", "a")
    
    min_dt = 1.e-4
    max_dt = 1.e3
    
    bin_fac = 2.
    nbins = bin_fac*ceil(log(max_dt/min_dt)/log(2.))
    
    ldt_min = log(min_dt)/log(2.)
    ldt_max = log(max_dt)/log(2.)
    
    dta = 2**(ldt_min+(ldt_max-ldt_min)*arange(nbins)/(nbins-1.))
    nbins=nbins-1
    dta1=shift(dta,1)
    dta = dta[0:nbins] ; dta1 = dta1[0:nbins]
    sum1=zeros((nbins),dtype='float32') ; sum2=zeros((nbins),dtype='float32')
    sum3=zeros((nbins),dtype='float32')
    pspec0=zeros((nbins),dtype='float32') ; pspec=zeros((nbins),dtype='float32') 
    dpspec=zeros((nbins),dtype='float32') ; nterms=zeros((nbins),dtype='float32')
    min_dta=dta.max() ; max_dta=0.
    #file_input = "00147478_bat_fine_wlc.txt"
    """
    00261664_bat_fine_wlc.txt
    00116116_bat_fine_wlc.txt
    00147478_bat_fine_wlc.txt
    00148225_bat_fine_wlc.txt
    00177408_bat_fine_wlc.txt
    00202035_bat_fine_wlc.txt
    00217805_bat_fine_wlc.txt
    
    """
    #file_input = "00310219_bat_fine_wlc.txt"
    #file_input = "00217805_bat_fine_wlc.txt"
    cv_chi2 = loadtxt('/Users/Vahid/python_codes/chi2_criticVal.txt',dtype='float32')
    dof = cv_chi2[:,0]
    cv = cv_chi2[:,1]
    
    DATA = loadtxt(file_input, dtype=float32)
    if (DATA.size > 0):
        t = DATA[:,0]; dt = DATA[:,1]; rate = DATA[:,2]; drate = DATA[:,3]
        lrate = log(rate)
        dlrate = drate/rate
        
        # Denoising the data
        lrate = haar_denoise.haar_denoise(lrate,dlrate)
        #
        
        nrepl=1 
        dt00=t.max()-t.min()
        dt0=dt ; t0=t ; lrate0=lrate ; dlrate0=dlrate
        for k in xrange(0,nrepl):
            t=concatenate((t,t0+(k+1)*dt00)) ; dt=concatenate((dt,dt0))
            lrate=concatenate((lrate,lrate0)) ; dlrate=concatenate((dlrate,dlrate0))
        
        # get the difference of every two points
        resl = haar_nondec.haar_nondec(t,lrate,dlrate,16.)    
        resl2 = asarray(resl)
        # return delta_t,tav,wav,dwav0,dwav ***from IDL***
        delta_t = resl2[0,:]; tav = resl2[1,:]
        wav = resl2[2,:]; dwav0 = resl2[3,:]; dwav = resl2[4,:]
        diff2 = wav**2 ; diff_var=dwav**2*(nrepl+1.)/bin_fac ; diff_var0=dwav0**2
        
        for j in xrange(0,int(nbins)):
            h1 = where(delta_t >= dta[j]); h2 = where((delta_t < dta1[j])*(diff_var > 0))
            h11 = asarray(h1); h22 = asarray(h2)
            h = intersect1d(h11,h22)
            nh = size(h)
            #h=where(delta_t >= dta[j] and delta_t < dta1[j] and diff_var > 0,nh)
            if (nh > 1):
                sum1[j] = sum(diff2[h]/diff_var0[h])
                sum2[j] = sum(1./diff_var[h])
                sum3[j] = sum(1./diff_var0[h])
                nterms[j] = nh
                if (dta[j] < min_dta): min_dta=dta[j]
                if (dta1[j] > max_dta): max_dta=dta1[j]
        
        #Sum2 = dpspec # Sum2 = 1/nterms * SUM_j,s 1/dwav^2_j,s
        
        
        g1 = where(sum3 > 0); g = asarray(g1)
        ng = size(g)
        if (ng > 0):
            pspec[g] = (sum1[g]/sum3[g])
            pspec0[g] = nterms[g]/sum3[g]
            dpspec[g] = sqrt(2.)/sqrt(sum2[g]*sum3[g]/nterms[g])
        
        
        # maybe we can ignore this part!    
        #g = where((abs(pspec - pspec0) < 2.*dpspec)*(dpspec > 0)) ;# g1 = asarray(g)
        g = where((abs(pspec) < 2.*dpspec)*(dpspec > 0)) ;# g1 = asarray(g)
        a = 1.
        ng = size(g)
        """
        if (ng > 3):
            a = sum(pspec[g]*pspec0[g]/(dpspec[g]**2)) / sum((pspec0[g]/dpspec[g])**2)
            pspec0[g] = pspec0[g]*a        
        
        print ' a:',a
        """
        
