import numpy as np
import matplotlib
import os
# Enables saving plots over ssh
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import argparse as ap
import scipy as sp
import scipy.constants as c
import scipy.integrate

def f_sun(Eg):
    Ts = 5777 #Kelvin
    a = Eg/(c.k*Ts)
    b = np.inf
    def integrand(x):
        return x**2/(np.exp(x)-1)
    return (integrand,a,b)

def f_amb(Eg):
    Ta = 300 #Kelvin
    a = Eg/(c.k*Ta)
    b = np.inf
    def integrand(x):
        return x**2/(np.exp(x)-1)
    return (integrand,a,b)

def f_cell(Eg,V):
    Tc = 300 #Kelvin
    A = np.exp(-(c.e*V)/(c.k*Tc))
    a = Eg/(c.k*Tc)
    b = np.inf
    def integrand(x):
        return x**2/(A*np.exp(x)-1)
    return (integrand,a,b)

def inc_power(Ts,Fs,Ta,Fa):
    coeff = (2*c.k**4)/(c.h**3*c.c**2)
    func = lambda x: x**3/(np.exp(x)-1) 
    intval = sp.integrate.quad(func,0,np.inf)[0]
    sun_pwr = Fs*Ts**4*intval
    amb_pwr = (1-Fs/Fa)*Ta**4*intval 
    return coeff*(sun_pwr+amb_pwr)

def jv(Eg,dist_fact,plot=True):
    Eg_ev = Eg
    Eg = Eg*c.e
    # Calculate geometric params
    sun_R = 695700*1000
    sun_D = 1.496E11/dist_fact
    theta_s = np.arctan(sun_R/(sun_D/dist_fact))
    Fs = np.pi*np.sin(theta_s)**2
    Fc = Fa = np.pi 
    Ts = 5777
    Ta = 300
    # Get incident power
    inc_pwr = inc_power(Ts,Fs,Ta,Fa)
    # Now calculate J by sweeping through V
    V_val = 0.0
    cross = False
    V = np.array([])
    J = np.array([])
    coeff = (2*c.e*c.k**3)/(c.h**3*c.c**2)
    T1 = sp.integrate.quad(*f_sun(Eg))[0]
    T2 = sp.integrate.quad(*f_amb(Eg))[0]
    while not cross:
        T3 = sp.integrate.quad(*f_cell(Eg,V_val))[0]
        #print("T1 = ",T1)
        #print("T2 = ",T2)
        #print("T3 - ",T3)
        jval = coeff*(Fs*Ts**3*T1 + (1-Fs/Fa)*Ta**3*T2 - Fc*Ta**3*T3)
        if jval < 0:
            cross = True
        else:
            J = np.append(J,jval)
            V = np.append(V,V_val)
        V_val += .01

    #V = np.arange(0,.6,.01)
    #J = np.zeros_like(V)
    #for i in range(V.size):
    #    coeff = (2*c.e*c.k**3)/(c.h**3*c.c**2)
    #    T1 = sp.integrate.quad(*f_sun(Eg))[0]
    #    T2 = sp.integrate.quad(*f_amb(Eg))[0]
    #    T3 = sp.integrate.quad(*f_cell(Eg,V[i]))[0]
    #    #print("T1 = ",T1)
    #    #print("T2 = ",T2)
    #    #print("T3 - ",T3)
    #    jval = coeff*(Fs*Ts**3*T1 + (1-Fs/Fa)*Ta**3*T2 - Fc*Ta**3*T3)
    #    J[i] = jval
    #J = J[J>0]
    #V = V[0:J.size]
    pwr_density = V*J
    conv_eff = pwr_density/inc_pwr
    if plot:
        fig1 = plt.figure(1)
        # Current and Power density
        ax1 = fig1.add_subplot(1,1,1)
        ax1.set_title('J(V) curve for Eg = %.3f eV'%Eg_ev)
        ax1.set_xlabel('Cell Voltage [V]')
        ax1.set_ylabel('Current Density J [A/m^2]')
        ax1.plot(V,J,'ro-',label='Current Density')
        ax1.plot(V,pwr_density,'bo-',label='Power Density')
        # Conversion efficiency
        ax2 = ax1.twinx()
        ax2.plot(V,conv_eff,'go-',label='Conversion\nEfficiency')
        ax2.set_ylabel('Conversion Efficiency')
        ax1.legend(loc='lower right')
        ax2.legend(loc='center left')
        #matplotlib.ticker.FuncForma
        #ax.get_xaxis().set_major_formatter(
        plt.show(fig1)
    return V,J,pwr_density,conv_eff

def plot_integrand(Eg):
    Eg_ev = Eg
    Eg = Eg*c.e
    Tc = 300 #Kelvin
    V = 0.7
    A = np.exp(-(c.e*V)/(c.k*Tc))
    a = Eg/(c.Boltzmann*Tc)
    b = np.inf
    def integrand(x):
        return x**2/(A*np.exp(x)-1)
    x = np.arange(0,100,1)
    y = integrand(x)
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1)
    ax.set_title('J(V) curve for Eg = %.3f eV'%Eg_ev)
    ax.set_xlabel('Integrand Value')
    ax.set_ylabel('Energy (kind of)')
    ax.plot(x,y,'ro-')
    #matplotlib.ticker.FuncForma
    #ax.get_xaxis().set_major_formatter(
    plt.show(fig1) 
    return (integrand,a,b)

def band_sweep(arglist):
    start = float(arglist[0])
    end = float(arglist[1])
    step = float(arglist[2])
    bands = np.arange(start,end,step)
    effs = np.zeros_like(bands)
    for i in range(bands.size):
        data = tuple(jv(bands[i],1,False))
        eff = np.amax(data[-1])
        effs[i] = eff
    fig2 = plt.figure(1)
    ax3 = fig2.add_subplot(1,1,1)
    ax3.set_title('Max Conversion Efficiency vs. Band Gap')
    ax3.set_xlabel('Band Gap [eV]')
    ax3.set_ylabel('Conversion Efficiency')
    ax3.plot(bands,effs,'bo-')
    plt.show(fig2)

def sun_sweep(sunvals,bandvals,colorsMap='jet'):
    start = float(bandvals[0])
    end = float(bandvals[1])
    step = float(bandvals[2])
    bands = np.arange(start,end+step,step)
    start = float(sunvals[0])
    end = float(sunvals[1])
    step = float(sunvals[2])
    suns = np.arange(start,end+step,step)
    effs = np.zeros((suns.size,bands.size))
    for i in range(suns.size):
        for j in range(bands.size):
            data = tuple(jv(bands[j],suns[i],False))
            eff = np.amax(data[-1])
            effs[i,j] = eff
    fig2 = plt.figure(1)
    ax3 = fig2.add_subplot(1,1,1)
    ax3.set_title('Conversion Efficiency Heatmap')
    ax3.set_xlabel('Band Gap [eV]')
    ax3.set_ylabel('Distance Reduction Factor')
    ax3.set_xlim((bands[0],bands[-1]))
    ax3.set_ylim((suns[0],suns[-1]))
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=np.amin(effs), vmax=np.amax(effs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    ax3.pcolormesh(bands, suns, effs,cmap=cm,norm=cNorm,alpha=.5)
    scalarMap.set_array(effs)
    cb = fig2.colorbar(scalarMap)
    cb.set_label('Conversion Efficiency')
    plt.show(fig2)
    
def main():
    parser = ap.ArgumentParser(description="""Performs detailed balance J(V) and efficiency calculations for a single
    band solar cell. Assumes only radiative recombination""")
    parser.add_argument('-bg','--band_gap',type=float,help="""The bandgap to calculate in eV""")
    parser.add_argument('-bs','--band_sweep',nargs=3,help="""Perform JV calculations and plot""")
    parser.add_argument('-ss','--sun_sweep',nargs=3,help="""Perform JV calculations and plot""")
    args = parser.parse_args()
  
    #plot_integrand(args.band_gap)
    if args.band_gap:
        jv(args.band_gap,1)
    if args.band_sweep and not args.sun_sweep: 
        band_sweep(args.band_sweep)
    if args.sun_sweep and args.band_sweep:
        sun_sweep(args.sun_sweep,args.band_sweep)
if __name__ == '__main__':
    main()
