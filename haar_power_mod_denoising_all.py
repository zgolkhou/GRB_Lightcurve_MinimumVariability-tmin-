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
        
        snr = 3.0
        cts = sum1  #-nterms*a 
        error = sqrt(sum3*2*nterms/((sum2<1.).choose(sum2,1))) ; dt = sum3
        
        #rate_rebin,dta,dta1,dt,cts,error,snr,0,2.,0.,ii
        rateRebin1 = rate_rebin.rate_rebin(dta,dta1,dt,cts,error,snr,0.,2.,0)
        rateRebin = asarray(rateRebin1)
        dta = rateRebin[0]; dta1 = rateRebin[1]; dt = rateRebin[2]
        cts = rateRebin[3]; error = rateRebin[4]; index = rateRebin[5]
        #
        pspec = cts/((dt<1.).choose(dt,1)) ; dpspec = error/((dt<1.).choose(dt,1))
        #do_rebin,sum3,ii
        doRebin1 = do_rebin.do_rebin(sum3,index)
        doRebin = asarray(doRebin1)
        sum3 = doRebin
        #
        #do_rebin,nterms,ii
        doRebin2 = do_rebin.do_rebin(nterms,index)
        doRebin_nterms = asarray(doRebin2)
        nterms = doRebin_nterms
        
        pspec0  = a*nterms/((sum3<1.).choose(sum3,1))
        cts = cts/((nterms<1.).choose(nterms,1))
        #
        #do_rebin,sum2,ii
        doRebin_sum2 = do_rebin.do_rebin(sum2,index)
        doRebin_sum2_2 = asarray(doRebin_sum2)
        sum2_2 = doRebin_sum2_2
        sum2_2 = sum2_2/((nterms<1.).choose(nterms,1))
        
        nsig = snr
        g0 = where((pspec > nsig*dpspec)*(dpspec>0))
        #g00 = asarray(g0)
        ng0 = size(g0)
        if (ng0>1):
            g00 = where(dpspec > 0)
            #g02 = asarray(g01);
            ng0 = size(g00)
            #
            all_t = 0.5*(dta[g00]+dta1[g00]); all_dt = 0.5*(dta1[g00]-dta[g00]) 
            all_sig = pspec[g00]; all_err0 = dpspec[g00]
            all_err_chk = (1.+2*cts[g00]) 
            all_err = all_err0*sqrt(((all_err_chk<0).choose(all_err_chk,0))) #check for the argument being>0
            all_sig0 = pspec0[g00]
            sum2_2_2 = sum2_2[g00]
            g1=where(all_sig < nsig*all_err0)
            #g1 = asarray(g1)
            ng = size(g1)
            #
            g2 = where(all_sig >= nsig*all_err0)
            #g2 = asarray(g2)
            ng2 = size(g2)
            if (ng > 0): # check for the arguments >0
                all_sig_chk = all_sig[g1]+nsig*all_err0[g1]       
                all_sig[g1] = sqrt(((all_sig_chk<0).choose(all_sig_chk,0)))
                all_err[g1] = 0.
                   
            if (ng2 > 0):
                # fitting mu_0 using chi2 minimization!
                pspec_p = all_sig[g2]
                dpspec_p = all_err[g2]
                tau_time = all_t[g2]
                sum22 = sum2_2_2[g2]
                #
                #        
                all_sig[g2] = sqrt(all_sig[g2])
                #all_err[g2] = 0.5*all_err[g2]/sqrt(all_sig0[g2])
                all_err[g2] = 0.5*all_err[g2]/sqrt( all_sig[g2] )
                #
                miny=min(all_sig[g2])/2.
                ##subplot(111, xscale="log", yscale="log")
                #axis = [min_dta/2.,max_dta*2., miny,max(append(all_sig([g2],1.)))*2]
                #errorbar(all_t, all_sig, all_dt, all_err,'r.-')
                #
                """
                ax = plt.subplot(111)
                ax.set_xscale("log", nonposx='clip')
                ax.set_yscale("log", nonposy='clip')
                plt.errorbar(all_t, all_sig, xerr=all_dt, yerr=all_err, fmt='.k')
                ax.set_xlim((min_dta/2.,max_dta*2.))
                maxy = max(append(all_sig[g2],1.))*2
                ax.set_ylim((miny, maxy))
                ax.set_title('Title')
                ax.set_xlabel(r'$\mathrm{\Delta T}$  [s]', fontsize=12)
                ax.set_ylabel(r'Flux Variation  $\mathrm{\sigma_{X,\Delta t}}$  [%]', fontsize=12)
                #ax.set_text(0.05, 0.9, 'Text goes here',
                #        fontsize=14, transform=pl.gca().transAxes,
                #        ha='left', va='bottom')
                #plt.show()
                #       
                xx = array([1.e-9,1.e9])
                for i in xrange(int(log10(min_dt)*2.-4.), int(log10(max_dt)*2)):
                    plt.plot(xx, miny*xx*exp(-i*log(10.)/2.),'c:', markersize=6)
                #
                if (ng > 0):
                    plt.plot(all_t[g1], all_sig[g1], 'bv')
                #
                plt.plot(xx,xx,'r-.')
                base1 = file_input+"_fluxVariance.png"
                base_address = '/home/zach/project_wavelet/Flux/'
                plt.savefig(base_address+base1,format='png')
                plt.clf()
                #plt.savefig('/Users/Vahid/Desktop/plots/file_input.png',format='png')
                #plt.savefig('/Users/Vahid/untitled/PLOTS/testplot.pdf',format='pdf')
                #Image.open('/Users/Vahid/untitled/PLOTS/testplot.png').save('/Users/Vahid/untitled/PLOTS/testplot.jpg','JPEG')
                #plt.show()
                """
                
                 
                #   
                # fitting mu0:
                #import mu0_minimize_CHI2_fmin
                MU0, CHI2 = mu0_minimize_CHI2_fmin.mu0_minimize_CHI2_fmin(pspec_p,dpspec_p,tau_time,sum22)
                #
                #prnt_chi2 = file_input+'  : '+str(CHI2)
                #text_file2.write("%s,\n"%prnt_chi2)
                prnt_chi3 = file_input
                text_file3.write("%s\n"%prnt_chi3)
                text_file2.write("%s,\n"%str(CHI2))
                prnt_tauMIN = file_input+':'+str(tau_time[0])
                text_file4.write("%s\n"%prnt_tauMIN)
                prnt_tauArray = file_input+':'+str(tau_time)
                text_file5.write("%s,\n"%prnt_tauArray)
                
                ####plt.figure(2)
                ###tau2_time = tau_time[1:]
                ###tau3_time = 0.5*(tau2_time[1:]+tau2_time[:-1])
                #prb = exp(-0.5*diff(CHI2))
                #plt.plot(tau3_time,prb,'bD-')
                ###chi2_diffTest = sqrt(diff(CHI2))
                ####plt.plot(tau3_time,chi2_diffTest,'r*--')
                ###base2 = file_input+"_CHI2.png"
                ###base_address = '/home/zach/project_wavelet/CHI2/'
                ###plt.savefig(base_address+base2,format='png')
                ###plt.clf()
                """
                whr =where(diff(sign(mrg - prb))!=0)
                sz_whr = size(whr)
                if (sz_whr > 0):
                    y1_1 = prb[whr[0]]
                    y1_2 = prb[whr[0]+1]
                    y2_1 = mrg[whr[0]]
                    y2_2 = mrg[whr[0]+1]
                    t_1 = tau3_time[whr[0]]
                    t_2 = tau3_time[whr[0]+1]
                    t_cross = ((t_2-t_1)*(y1_1-y2_1)-t_1*((y1_2-y1_1)-(y2_2-y2_1)))/((y2_2-y2_1)-(y1_2-y1_1))
                else:
                    t_cross = 'NA'
                """
                            
                wh_cv = where(CHI2 <= cv[:size(CHI2)])
                if (size(wh_cv) > 0):
                    ref = arange(size(wh_cv))
                    prnt_cvSize = file_input+':'+str(size(wh_cv))
                    text_file6.write("%s\n"%prnt_cvSize)
                    diff_ref = wh_cv - ref
                    wh_ref = where(diff_ref[0] != 0)
                    if (size(wh_ref[0]) == 0):
                        CHI2 = [CHI2[x] for x in wh_cv[0]]
                        #tau_time = [tau_time[y] for y in wh_cv[0]]
                        tau_time = tau_time[:max(wh_cv[0])+2]
                    else:
                        wh_indx = min(wh_ref[0])
                        CHI2 = CHI2[:wh_indx]
                        tau_time = tau_time[:wh_indx+1]
                        
                    CHI2 = array(CHI2)
                    tau_time = array(tau_time)
                    tau2_time = tau_time[1:]
                    tau3_time = 0.5*(tau2_time[1:]+tau2_time[:-1])
                    
                    chi2_diffTest = sqrt(abs(diff(CHI2)))
                    
                    if (chi2_diffTest.size > 0):
                        thrshld = 2.
                        sigma2_cl = where(chi2_diffTest >= thrshld)
                        sigma2_cl2 = sigma2_cl[0]
                        if (sigma2_cl2.size == 0):
                            t_min = tau3_time[-1]
                            #prnt = file_input+'  : '+str(t_min)+'     D     '+str(chi2_diffTest[-1])
                            prnt = file_input+':'+str(t_min)+'  '+str(CHI2[-1]/(CHI2.size-1))+'  '+str(CHI2[-1])+'  '+str(CHI2.size-1)+' -1 '+str(chi2_diffTest[-1])
                            text_file1.write("%s\n"%prnt)
                        else:
                            sigma2_indx = sigma2_cl2[0]
                            if (sigma2_indx > 0):
                                y_1 = chi2_diffTest[sigma2_indx-1]
                                y_2 = chi2_diffTest[sigma2_indx]
                                t_1 = tau3_time[sigma2_indx-1]
                                t_2 = tau3_time[sigma2_indx]
                                t_min = (thrshld-y_2)/(y_2-y_1)*(t_2-t_1)+t_2
                                CHI2_0_dof = CHI2[sigma2_indx]/(sigma2_indx)
                                prnt = file_input+':'+str(t_min)+'  '+str(CHI2_0_dof)+'  '+str(CHI2[sigma2_indx])+'  '+str(sigma2_indx)+' 0 '+' 0 '
                                #prnt = file_input+'  : '+str(t_min)
                                text_file1.write("%s\n"%prnt)
                            else:
                                t_min = tau3_time[sigma2_indx]
                                prnt = file_input+':'+str(t_min)+'  '+str(CHI2[0])+'  '+' 0 '+'   ' +' 0 '+'  1  '+str(chi2_diffTest[0])
                                #prnt = file_input+'  : '+str(t_min)+'     U    '+str(chi2_diffTest[0])
                                text_file1.write("%s\n"%prnt)
                                
                        print 't_min = ',t_min
                    
    text_file1.close()    
    text_file2.close()
    text_file3.close()
    text_file4.close()
    text_file5.close()
    text_file6.close()
        #print 't_min = ',t_min

    return
