#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jun 7 12:33:17 2021
### Change
@author: adameshel
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import sys
import math
from autocorr_functions import *

class Autocorr():
    def __init__(self, df, bw, cutoff_distance_km=90.0):
        """bw - bandwidth in km"""
        bw = bw * 1e3 # convert to meters
        cutoff_distance_km = cutoff_distance_km * 1e3 # convert to meters
        if 'L' in df:
            xMax = np.max([df.xa.max(), df.xb.max()])
            xMin = np.min([df.xa.min(), df.xb.min()])
            yMax = np.max([df.ya.max(), df.yb.max()])
            yMin = np.min([df.ya.min(), df.yb.min()])
        else:
            xMax = np.max(df.x.max())
            xMin = np.min(df.x.min())
            yMax = np.max(df.y.max())
            yMin = np.min(df.y.min())

        p_prep = np.array( df[['x','y','z']] )
        max_num_of_rg_in_row = len(p_prep[0,2])
        ## A loop for defining the VRGs per link where the number  
        ## of VRGs per link is not the same
        # if type(p_prep[0,2]) is tuple:
        #     for row in range(len(p_prep[:,0])):
        #         num_of_rg_in_row = len(p_prep[row,0])
        #         p_prep[row,2] = p_prep[row,2][:num_of_rg_in_row]
        # else:
        #     for row in range(len(p_prep[:,0])):
        #         num_of_rg_in_row = 1
        #         p_prep[row,2] = p_prep[row,2]



        for row in range(len(p_prep[:,0])):
            num_of_rg_in_row = len(p_prep[row,0])
            p_prep[row,2] = p_prep[row,2][:num_of_rg_in_row]
                
            
        p = np.zeros([len(p_prep[:,0]) * max_num_of_rg_in_row,
                    len(p_prep[0,:])])
        link_num_prep = []
        link_L_prep = []
        p_row = 0
        element = 0
        for row in range(len(p_prep[:,0])):
            if row != 0:
                p_row = p_row + element +1
            for col in range(len(p_prep[0,:])):
                for element in range(len(p_prep[row,2])):
                    # import pdb; pdb.set_trace()
                    p[p_row+element,col] = p_prep[row,col][element]
        for cut_point in range(len(p[:,0])): 
            if p[cut_point,0] == 0: 
                break
        if len(p[:,0]) == (cut_point + 1):
            p_filtered = p
        else:
            p_filtered = np.delete(p, 
                                range(cut_point,len(p_prep[:,0]) *\
                                        max_num_of_rg_in_row), 
                                0)
        if 'L' in df:
            for row in range(len(p_prep[:,0])):
                for element in range(len(p_prep[row,2])):
                    link_num_prep.append(df['Link_num'][row])
                    link_L_prep.append(df['L'][row])

        self.df_p = pd.DataFrame(p_filtered, columns=['x','y','z'])
        if 'L' in self.df_p:
            self.df_p['l_num'] = link_num_prep
            self.df_p['L'] = link_L_prep

        self.hs = np.arange(bw, 
                    np.min([np.max([xMax-xMin,yMax-yMin]), 
                    cutoff_distance_km]), 
                    bw*2.0)
        min_dist = [0,0]
        p = self.df_p[['x','y','z']].values
        self.ac = self._AC( p, self.hs, bw )


    def __call__(self, optimize=True):
        '''
        Choose method by which you wish to find  the exponential and 
        multiplication parameters- alpha_L and beta_L.
        
        optimize=True: scipy optimize acf_original to data.
        optimize=False: beta_L is the max value and alpha_L is the 
        value of h at which the function lost 95%.
        '''
        ## Choose the nugget before optimizing
        if len(self.hs) > 15:
            mins = int(len(self.ac[1])/6)
            self.nugget = np.nanmedian(np.sort(self.ac[1])[:mins])
        else:
            self.nugget = np.min(self.ac[1])

        # ihc = np.sum(self.ac[1] >= np.nanmax(self.ac[1]) * 0.05) -1
        # self.nugget = np.nanmedian(self.ac[1][ihc:])
        # self.ac[1]

        # self.nugget = np.min(self.ac[1])
        if optimize==True:
            self.magnitude_beta = 10 ** (int(np.log10(np.var(self.ac[1]))))
            self.magnitude_alpha = 10 ** (int(np.log10(np.nanmean(self.hs))))
            self.ac[0] = self.ac[0] / self.magnitude_alpha
            self.ac[1] = self.ac[1] / self.magnitude_beta
            self.nugget = self.nugget / self.magnitude_beta
            # import pdb; pdb.set_trace()
            popt, _ = curve_fit(
                f=acf_original, 
                xdata=self.ac[0],
                ydata=self.ac[1]-self.nugget
            )
            self.alpha_L, self.beta_L = popt
            self.alpha_L  = self.alpha_L * self.magnitude_alpha
            self.beta_L = self.beta_L * self.magnitude_beta
            self.ac[0] = self.ac[0] * self.magnitude_alpha
            self.ac[1] = self.ac[1] * self.magnitude_beta
            self.nugget = self.nugget * self.magnitude_beta
        else:
            # Max value of ACF
            self.beta_L = self.ac[1][0]
            # Correlation distance as h where 
            #the value of ACF decreases by 95%
            self.alpha_L = \
                self.hs[np.sum((self.ac[1] - self.nugget) >= \
                    (np.nanmax(self.ac[1] - self.nugget) * 0.05)) -1]
        
    def _ACh( self, P, h, bw ):
        '''
        Experimental autocorrelation function for a single lag
        '''
        p_d = squareform( pdist( P[:,:2] ) )
        N = p_d.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i,N):
                if( p_d[i,j] >= h-bw )and( p_d[i,j] <= h+bw ):
                    Z.append( ( P[i,2] * P[j,2] ) )
        if len(Z)==0:
            return -1
        return np.sum( Z ) / ( len( Z ) )
    
    def _AC( self, P, hs, bw ):
        '''
        Experimental autocorrelation function for a collection of lags
        '''
        ac = list()
        for h in hs:
            ac.append( self._ACh( P, h, bw ) )
        ac = [ [ hs[i], ac[i] ] for i in range( len( hs ) ) if ac[i] != -1 ]
        return np.array( ac ).T


            