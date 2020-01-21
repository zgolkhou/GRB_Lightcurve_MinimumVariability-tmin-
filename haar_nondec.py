from numpy import log,hstack,arange,round,unique,where,zeros,sqrt,cumsum,size,asarray,loadtxt,concatenate
import haar_denoise

def haar_nondec(file_input,nbins,dta,dta1,over_samp=16.,nrepl=1.,bin_fac=2.):
    """
    """
    DATA = loadtxt(file_input, dtype='float64')
    t = DATA[:,0]; dt = DATA[:,1]; rate = DATA[:,2]; drate = DATA[:,3]
    print '\n Data_Size =%d \n'%(size(t))
    lrate = log(rate) 
    dlrate = drate/rate

    # PAUSE
    #raw_input('press a key!')

    # Denoising the data
    lrate = haar_denoise.haar_denoise(lrate,dlrate); print "\n Denoising is Done! \n"
    #
    #tic2 = time.time()
    #print "\n elapsed_time[1] : ", tic2-tic1

    nrepl=1   #=1
    dt00=t.max()-t.min()
    dt0=dt ; t0=t ; lrate0=lrate ; dlrate0=dlrate
    for k in xrange(0,nrepl):
        t=concatenate((t,t0+(k+1)*dt00)) ; dt=concatenate((dt,dt0))
        lrate=concatenate((lrate,lrate0)) ; dlrate=concatenate((dlrate,dlrate0))

    #time = time.astype('float64')
    #data = data.astype('float64')
    #ddata = ddata.astype('float64')

    sum1=zeros((nbins),dtype='float64') ; sum2=zeros((nbins),dtype='float64')   
    sum3=zeros((nbins),dtype='float64') ; nterms=zeros((nbins),dtype='float64')
    min_dta=dta.max() ; max_dta=0.

    nmax = len(lrate)
    cx=cumsum(lrate); vx=cumsum(dlrate**2); ctt=cumsum(t)
    cx=hstack((0,cx)); vx=hstack((0,vx)); ctt=hstack((0,ctt))
    lscl_max=log(nmax)/log(2.)

    scales=2**( arange(lscl_max,dtype='float32') )

    if (over_samp<=0):
        over_samp=1.

    scales2=2.*round( 2**( arange(lscl_max*over_samp,dtype='float32')/over_samp)/2.  )
    scales=hstack((scales,scales2))
    scales.sort(); scales = unique( scales )

    s=where( (scales>0)*(2*scales<=nmax) )
    scales=scales[s].astype('int32')

    nscales=len(scales)
    #ntot=nscales*nmax
    ntot = cumsum(nmax-2*scales[:]+1)[-1]+1 # adding +1 for no/? reason!
    wav=zeros(ntot,dtype='float64')
    dwav=zeros(ntot,dtype='float64')
    delta_t=zeros(ntot,dtype='float64')
    tav=zeros(ntot,dtype='float64')
    dwav0=zeros(ntot,dtype='float64')
    ii0 = arange(nmax,dtype='int32')

    nn=0
    for k in xrange(nscales):
        scl=scales[k]
        nn1 = nn+nmax-2*scl
        ii = ii0[0:nmax-2*scl+1]
        wav[nn:nn1+1] = ( cx[ii+2*scl] - 2*cx[ii+scl] + cx[ii] )/scl
        dwav0[nn:nn1+1] = sqrt( vx[ii+2*scl] - vx[ii] )/scl
        dwav[nn:nn1+1] = dwav0[nn+ii]*sqrt(2.*over_samp*scl)
        delta_t[nn:nn1+1] = ( ctt[ii+2*scl] - 2*ctt[ii+scl] + ctt[ii] )/scl
        tav[nn:nn1+1] = ( ctt[ii+2*scl] - ctt[ii] )/scl/2.
        nn+=nmax-2*scl+1

    wav=wav[0:nn]; dwav=dwav[0:nn]; delta_t=delta_t[0:nn]; tav=tav[0:nn]; dwav0=dwav0[0:nn]
    #print "\n size(tav)befor : \n\n",size(tav)
    g=where( (delta_t>0)*(dwav>0) )
    #print "\n size(g) : \n\n",size(g)
    ###wav=wav[g]; dwav=dwav[g]; dwav0=dwav0[g]; delta_t=delta_t[g]; tav=tav[g]    <-------------
    #print "\n size(tav) : \n\n",size(tav)
    delta_t=delta_t[g]; tav=tav[g]
    diff2 = (wav[g])**2 ; diff_var=((dwav[g])**2)*(nrepl+1.)/bin_fac ; diff_var0=(dwav0[g])**2
    
    for j in xrange(0,int(nbins)):
        h = where((delta_t >= dta[j])*(delta_t < dta1[j])*(diff_var > 0))[0]
        #h1 = where(delta_t >= dta[j]); h2 = where((delta_t < dta1[j])*(diff_var > 0))
        #h11 = asarray(h1); h22 = asarray(h2) # TIME KILLER LINE
        #h = intersect1d(h11,h22) # TIME KILLER LINE
        nh = size(h)
        #h=where(delta_t >= dta[j] and delta_t < dta1[j] and diff_var > 0,nh)
        if (nh > 1):
            sum1[j] = sum(diff2[h]/diff_var0[h])
            sum2[j] = sum(1./diff_var[h])
            sum3[j] = sum(1./diff_var0[h])
            nterms[j] = nh
            if (dta[j] < min_dta): min_dta=dta[j]
            if (dta1[j] > max_dta): max_dta=dta1[j]
    
    return sum1,sum2,sum3,nterms,min_dta,max_dta
