# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 23:07:52 2013

@author: V. Zach
"""
from numpy import *
from matplotlib import pyplot as plt
from scipy import stats
from KaplanMeier import KaplanMeier as KM
from KaplanMeier_2 import KaplanMeier_2 as KM2

'''Logrank hypothesis test, comparing the survival times for two different datasets'''

# the following file is just an example.
input_file = "/Users/Vahid/Desktop/DATA_Christmas2013/merge_tmin_temp_spec_11_18_2013_rt45.txt"

DATA = loadtxt(input_file, dtype='float64')

save_plt = 0
save_plt_2 = 0
thresh = 0.1
#

grb_all = DATA[:,0]; tmin_all = DATA[:,1] # bin=2, over_sample = 16, chi2_diff (v2)
lev_all = DATA[:,2]; tw_all = DATA[:,3]
tt_all = DATA[:,4]; chi2_nu_all = DATA[:,5] 
dof_all = DATA[:,6]; pval_all = DATA[:,7] 
chi2_diffT_all = DATA[:,8]; tmax_all = DATA[:,9]; Lp_all = DATA[:,10]

grb_all2 = DATA[:,11]; t90_all = DATA[:,12]; t90_dev_all = DATA[:,13]

grb_all3 = DATA[:,14]; snr_all = DATA[:,15]; z_all = DATA[:,16]
Ep_all = DATA[:,17]; Epm_all = DATA[:,18]; Epp_all = DATA[:,19]
Sbol_all = DATA[:,20]; Sbolm_all = DATA[:,21]; Sbolp_all = DATA[:,22]

grb_all4 = DATA[:,23]; rt45_all = DATA[:,24]; rt45_all_dev = DATA[:,25]

pck = where((pval_all > thresh)*(lev_all==1))[0]  #*(t90_all>=3)
med_1 = median(tmin_all[pck]) 

#med_2 = median(tmin_all[pck]/(z_all[pck]+1.))
#----------------------------------------------------
pck2 = where((pval_all <= thresh)*(lev_all==1))[0]

len_mat = size(where(lev_all!=2))
km_data = zeros((len_mat,2))

km_data[:size(pck),0] = (tmin_all[pck]) 
#med_1 = median(tmin_all[pck]) 
km_data[size(pck):len_mat,0] = (tmin_all[pck2])
km_data[size(pck):len_mat,1] = ones(size(pck2))

km_data[:,0] = log10(km_data[:,0])
sdd = km_data[:,0].max()
km_data[:,0] = km_data[:,0].max() - km_data[:,0]
km_data = km_data[km_data[:,0].argsort()]

times_1 = km_data[:,0]
censored_1 = km_data[:,1]
atRisk_1 = arange(len(times_1),0,-1)
failures_1 = times_1[censored_1==0]
#---------------------------------------------------
#len_mat_2 = size(pck)
km_data_2 = zeros((size(pck),2))

km_data_2[:size(pck),0] = (tmin_all[pck])
#km_data[size(pck):len_mat,0] = (tmin_all[pck2])

km_data_2[:,0] = log10(km_data_2[:,0])
sdd_2 = km_data_2[:,0].max()
km_data_2[:,0] = km_data_2[:,0].max() - km_data_2[:,0]
km_data_2 = km_data_2[km_data_2[:,0].argsort()]

times_2 = km_data_2[:,0]
censored_2 = km_data_2[:,1]
atRisk_2 = arange(len(times_2),0,-1)
failures_2 = times_2[censored_2==0]
#---------------------------------------------------
failures = unique(hstack((times_1[censored_1==0], times_2[censored_2==0])))
num_failures = len(failures)
r1 = zeros(num_failures)
r2 = zeros(num_failures)
r  = zeros(num_failures)
f1 = zeros(num_failures)
f2 = zeros(num_failures)
f  = zeros(num_failures)
e1 = zeros(num_failures)
f1me1 = zeros(num_failures)
v = zeros(num_failures)

for ii in range(num_failures):
    r1[ii] = sum(times_1 >= failures[ii])
    r2[ii] = sum(times_2 >= failures[ii])
    r[ii] = r1[ii] + r2[ii]
    
    f1[ii] = sum(failures_1==failures[ii])
    f2[ii] = sum(failures_2==failures[ii])
    f[ii] = f1[ii] + f2[ii]
    
    e1[ii] = r1[ii]*f[ii]/r[ii]
    f1me1[ii] = f1[ii] - e1[ii]
    v[ii] = r1[ii]*r2[ii]*f[ii]*(r[ii]-f[ii]) / ( r[ii]**2 *(r[ii]-1) )

    O1 = sum(f1)
    O2 = sum(f2)
    E1 = sum(e1)
    O1mE1 = sum(f1me1)
    V = sum(v[:-1]) # <---- delete nan
    
chi2 = (O1-E1)**2/V
p = stats.chi2.sf(chi2, 1)

print('X^2 = {0}'.format(chi2))
if p < 0.05:
    print('p={0}, the two survival curves are signifcantly different.'.format(p))
else:
    print('p={0}, the two survival curves are not signifcantly different.'.format(p))
    
    
(p1, r1, t1, sp1,se1) = KM(km_data)
(failures,p,see) = KM2(km_data)
#
(p2, r2, t2, sp2,se2) = KM(km_data_2)
(failures_2,p_2,see_2) = KM2(km_data_2)

if 1:
    fig = plt.figure(1,(10,8))  #(10,8)  (6.5,3.5)  
    ax = fig.add_subplot(121)
    subplots_adjust(wspace=0.28) # 0.25
    ###plt.gca().invert_xaxis()
    p1, = ax.step((10**(sdd-failures)),p*size(p), where='pre',linewidth=1.5,color='green',alpha=0.4,label='gr'); plt.xscale('log')
    ##p2, = ax.plot((10**(sdd-failures)),(p+see)*size(p),color='w', linestyle='--', drawstyle='steps',alpha=0.4,label='re'); plt.xscale('log')
    ##ax.plot((10**(sdd-failures)),(p-see)*size(p),color='w', linestyle='--', drawstyle='steps',alpha=0.4); plt.xscale('log')
    p3, = ax.plot([2,2],[1,1],'b-',linewidth=3,label='dummy') # DUMMY LINE
    p4, = ax.plot([2,2],[1,1],'w-')    # DUMMY LINE
    p5, = ax.plot([2,2],[1,1],'r-',linewidth=1.5,alpha=0.4)  # DUMMY LINE
    p6, = ax.plot([2,2],[1,1],'k-',linewidth=1.5,alpha=0.4)  # DUMMY LINE

    #
    ax.set_xlim((0.01,max(10**(sdd-failures))))
    ax.set_ylim((0,size(p)+1))    
    #ax.set_title('Kaplan-Meier survival curve')
    ax.set_xlabel(r'$\Delta t_{min}\ (sec)$',fontweight='bold',fontdict={'fontsize':18})
    ax.set_ylabel(r'$N(>\Delta t_{min})$',fontweight='bold',fontdict={'fontsize':18})
    #ax.grid(True, which="both")
    left, width = .5, .45
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax.text(right, 0.5*(bottom+top), 'Fraction',color='black',size=16,
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)
       
    ax2 = ax.twinx()
    ax2.step((10**(sdd_2-failures_2)),p_2, where='pre',linewidth=3,color='blue'); plt.xscale('log')    
    #plt.plot([tmin.min(),tmin.max()],[0,0],'w-',linewidth=3)
    p_22 = p_2*1.; sdd_22 = sdd_2*1.; failures_22 = failures_2*1.
    #ax2 = ax.twinx()
    ax2.set_xlim((0.01,max(10**(sdd-failures))))
    ax2.plot([0.01,200],[0.5,0.5],'k:',alpha=0.5)
    ax2.plot([1.7853,1.7853],[0,1],'k:',alpha=0.5)
    ax2.plot([0.8956,0.8956],[0,1],'k:',alpha=0.5)
    ax2.plot([0.6373,0.6373],[0,1],'k:',alpha=0.5)
    ax2.plot([0.45145,0.45145],[0,1],'k:',alpha=0.5)
    #ax2.set_ylabel('Fraction',size=16,color='blue')
    
    lg = ax.legend((p3,p4,p1,p4,p5,p6), ('Fraction of all GRBs w/',' measured $\Delta t_{min}$ ','All, including upper limits',' via survival analysis','$T_{90}$ as upper-limit','$T_{r45}$ as upper-limit'),loc='upper left',handlelength=0.8,handletextpad=0.2)
    ###lg2 = plt.legend(('dfdf',''),loc='upper left')
    lg.draw_frame(False)
    ###lg2.draw_frame(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    #ax2.grid(True, which="both")
    if 0:
        plt.savefig('/Users/Vahid/Desktop/KM.png',format='png')
   
##################################################
pck = where((pval_all > thresh)*(lev_all==1)*(z_all !=0))[0]

#----------------------------------------------------
pck2 = where((pval_all <= thresh)*(lev_all==1)*(z_all !=0))[0]

len_mat = size(pck)+size(pck2) # size(where(lev_all!=2))
km_data = zeros((len_mat,2))

km_data[:size(pck),0] = (tmin_all[pck]/(z_all[pck]+1.)) # rest frame
med_2 = median(tmin_all[pck]/(z_all[pck]+1.))
km_data[size(pck):len_mat,0] = (tmin_all[pck2]/(z_all[pck2]+1.)) #rest frame
km_data[size(pck):len_mat,1] = ones(size(pck2))

km_data[:,0] = log10(km_data[:,0])
sdd = km_data[:,0].max()
km_data[:,0] = km_data[:,0].max() - km_data[:,0]
km_data = km_data[km_data[:,0].argsort()]

times_1 = km_data[:,0]
censored_1 = km_data[:,1]
atRisk_1 = arange(len(times_1),0,-1)
failures_1 = times_1[censored_1==0]
#---------------------------------------------------
#len_mat_2 = size(pck)
km_data_2 = zeros((size(pck),2))

km_data_2[:size(pck),0] = (tmin_all[pck]/(z_all[pck]+1.)) # rest frame
#km_data[size(pck):len_mat,0] = (tmin_all[pck2])
#km_data[size(pck):len_mat,1] = ones(size(pck2))

km_data_2[:,0] = log10(km_data_2[:,0])
sdd_2 = km_data_2[:,0].max()
km_data_2[:,0] = km_data_2[:,0].max() - km_data_2[:,0]
km_data_2 = km_data_2[km_data_2[:,0].argsort()]


times_2 = km_data_2[:,0]
censored_2 = km_data_2[:,1]
atRisk_2 = arange(len(times_2),0,-1)
failures_2 = times_2[censored_2==0]
#---------------------------------------------------
failures = unique(hstack((times_1[censored_1==0], times_2[censored_2==0])))
num_failures = len(failures)
r1 = zeros(num_failures)
r2 = zeros(num_failures)
r  = zeros(num_failures)
f1 = zeros(num_failures)
f2 = zeros(num_failures)
f  = zeros(num_failures)
e1 = zeros(num_failures)
f1me1 = zeros(num_failures)
v = zeros(num_failures)

for ii in range(num_failures):
    r1[ii] = sum(times_1 >= failures[ii])
    r2[ii] = sum(times_2 >= failures[ii])
    r[ii] = r1[ii] + r2[ii]
    
    f1[ii] = sum(failures_1==failures[ii])
    f2[ii] = sum(failures_2==failures[ii])
    f[ii] = f1[ii] + f2[ii]
    
    e1[ii] = r1[ii]*f[ii]/r[ii]
    f1me1[ii] = f1[ii] - e1[ii]
    v[ii] = r1[ii]*r2[ii]*f[ii]*(r[ii]-f[ii]) / ( r[ii]**2 *(r[ii]-1) )

    O1 = sum(f1)
    O2 = sum(f2)
    E1 = sum(e1)
    O1mE1 = sum(f1me1)
    V = sum(v[:-1])  # <---- delete nan
    
chi2 = (O1-E1)**2/V
p = stats.chi2.sf(chi2, 1)

print('X^2 = {0}'.format(chi2))
if p < 0.05:
    print('p={0}, the two survival curves are signifcantly different.'.format(p))
else:
    print('p={0}, the two survival curves are not signifcantly different.'.format(p))
    
    
(p1, r1, t1, sp1,se1) = KM(km_data)
(failures,p,see) = KM2(km_data)
#
(p2, r2, t2, sp2,se2) = KM(km_data_2)
(failures_2,p_2,see_2) = KM2(km_data_2)

if 1:
    ax = fig.add_subplot(122)
    ###plt.gca().invert_xaxis()
    p1, = ax.step((10**(sdd-failures)),p*size(p), where='pre',linewidth=1.5,color='green',alpha=0.4,label='gr'); plt.xscale('log')
    #plt.plot(failures,p,color='g',linestyle='--', drawstyle='steps',ms=5)
    ##p2, = ax.plot((10**(sdd-failures)),(p+see)*size(p),color='w', linestyle='--', drawstyle='steps',alpha=0.4,label='re'); plt.xscale('log')
    ##ax.plot((10**(sdd-failures)),(p-see)*size(p),color='w', linestyle='--', drawstyle='steps',alpha=0.4); plt.xscale('log')
    p3, = ax.plot([2,2],[1,1],'b-',linewidth=3,label='dummy') # DUMMY LINE
    p4, = ax.plot([2,2],[1,1],'w-')    # DUMMY LINE
    p5, = ax.plot([2,2],[1,1],'r-',linewidth=1.5,alpha=0.4)    # DUMMY LINE
    p6, = ax.plot([2,2],[1,1],'k-',linewidth=1.5,alpha=0.4)    # DUMMY LINE
    #
    ax.set_xlim((0.01,max(10**(sdd-failures))))
    ax.set_ylim((0,size(p)+1))    
    #ax.set_title('Kaplan-Meier survival curve')
    ax.set_xlabel(r'$\Delta t_{min}/(1+z)\ (sec)$',fontdict={'fontsize':18})
    #ax.set_ylabel(r'$N(>t_{min})$',fontdict={'fontsize':20})
    #ax.grid(True, which="both")
    left, width = .02, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax.text(left, 0.5*(bottom+top), r'$N(>\frac{\Delta t_{min}}{1+z})$',fontdict={'fontsize':18},
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)
    
    #3tmin = tmin_all[pck]
    #plt.figure(1,(14,8))
    #plt.subplot(121)
    #n, bins, patches = plt.hist(log10(tmin),bins=15,normed=1,histtype='step', lw=2, color='blue',cumulative=True)
    #n, bins, patches = plt.hist(tmin,bins=logspace(log10(tmin.min()),log10(tmin.max()),20),normed=1,histtype='step', lw=2, color='black',cumulative=True)
    ax2 = ax.twinx()
    ax2.step((10**(sdd_2-failures_2)),p_2, where='pre',linewidth=3,color='blue'); plt.xscale('log')    
    #plt.plot([tmin.min(),tmin.max()],[0,0],'w-',linewidth=3)
    
    ax2.plot([0.01,200],[0.5,0.5],'k:',alpha=0.5)
    ax2.plot([0.4618,0.4618],[0,1],'k:',alpha=0.5)
    ax2.plot([0.2437,0.2437],[0,1],'k:',alpha=0.5)
    ax2.plot([0.2153,0.2153],[0,1],'k:',alpha=0.5)
    ax2.plot([0.19167,0.19167],[0,1],'k:',alpha=0.5)

    #ax2 = ax.twinx()
    ax2.set_xlim((0.007,max(10**(sdd-failures))))
    ax2.set_ylabel('Fraction',size=16,color='black')
    
    lg = ax.legend((p3,p4,p1,p4,p5,p6), ('Fraction of all GRBs w/',' measured $\Delta t_{min}$ ','All, including upper limits',' via survival analysis','$T_{90}$ as upper-limit','$T_{r45}$ as upper-limit'),loc='upper left',handlelength=0.8,handletextpad=0.2)
    ###lg2 = plt.legend(('dfdf',''),loc='upper left')
    lg.draw_frame(False)
    ###lg2.draw_frame(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)    
    #ax2.grid(True, which="both")
    fig.tight_layout()
    if save_plt:
        #plt.savefig('/Users/Vahid/Desktop/KM2.png',format='png')
        plt.savefig('/Users/Vahid/Desktop/b6.pdf',format='pdf', bbox_inches='tight', pad_inches = 0.1)
    
    ##################################################
    plt.show()
