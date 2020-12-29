#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 09:43:34 2018

@author: adameshel
"""

'''
Implementation of GMZ algorithm.
Based on "Rain Rate Estimation Using Measurements From
Commercial Telecommunications Links"
by Oren Goldshtein, Hagit Messer and Artem Zinevich
'''
# from __future__ import print_function
import numpy as np
from collections import Counter


def apply_inverse_power_law(A, L, a, b):
    ''' variance of the attenuation measurement error.
    Formula (1) from paper.
    A - attenuation
    L - cml length in KM
    a, b - ITU power law parameters
    '''
    return (A/(a*L))**(1.0/b)


def apply_power_law(R, L, a, b):
    ''' variance of the attenuation measurement error.
    Formula (1) from paper.
    R - rain
    L - cml length in KM
    a, b - ITU power law parameters
    '''
    return (R**b)*a*L


def calc_rain_from_atten(df):
    '''calculate the average rain rate at each cml.
    df should contain the following columns:
    A - attenuation due to rain
    L - length of the cmls in KM
    a, b - ITU power law parameters
    '''
    df['R'] = df.apply(lambda cml: apply_inverse_power_law(cml['A'],
                                                           cml['L'],
                                                           cml['a'],
                                                           cml['b']),
                       axis=1)
    return df


def calc_atten_from_rain(df):
    '''calculate the attenuation from the avg. rain.
    df should contain the following columns:
    R - rain
    L - length of the cmls in KM
    a, b - ITU power law parameters
    '''
    df['A'] = df.apply(lambda cml: apply_power_law(cml['R'],
                                                   cml['L'],
                                                   cml['a'],
                                                   cml['b']),
                       axis=1)
    return df

def create_virtual_gauges(df, 
                          gauge_length=0.5,
                          num_gauges=None):
    ''' split each cml (a single row of df) into several virtual
    rain gauges.
    gauge_length - the distance between two gauges in KM. Not yot verified.
    num_gauges - number of virtual gauges per cml (overrides gauge_length)
    df should contain the following columns:
    xa - longitude of site A of the cml
    ya - latitude of site A of the cml
    xb - longitude of site B of the cml
    yb - latitude of site B of the cml
    L - length of the cml
    '''
    df = df.copy()
    x_gauges = []
    y_gauges = []
    z_gauges = []

    # split each cml into several virtual rain gauges
    for i, cml in df.iterrows():
        if num_gauges is None:
            num_gauges_along_cml = int(np.ceil(cml['L'] / float(gauge_length)))
#            vrgs_per_cml = 'ed' + str(round(gauge_length,2))
            vrgs_per_cml = 'ED' + str(int(gauge_length * 1e3))
        else:
            num_gauges_along_cml = int(num_gauges)
            vrgs_per_cml = num_gauges
        x, y = get_gauges_lon_lat(cml['xa'], cml['ya'],
                                  cml['xb'], cml['yb'],
                                  G=num_gauges_along_cml)

        # initial rain value for each gauge
        z = tuple([cml['R']] * num_gauges_along_cml)

        x_gauges.append(x)
        y_gauges.append(y)
        z_gauges.append(z)

    # add x, y locations of the virtual rain gauges of each cml
    df['x'] = x_gauges
    df['y'] = y_gauges

    # add initial z (rain rate) of each virtual rain gauge of each cml
    df['z'] = z_gauges
    return df, vrgs_per_cml


def get_gauges_lon_lat(a_lon, a_lat, b_lon, b_lat, G=2):
    """ Calculate and return longitude and latitude
    of G gauges along the cml """
    def GaugeCoords(t):
        return a_lon + t*(b_lon - a_lon), a_lat + t*(b_lat - a_lat)

    LonTuple = []
    LatTuple = []
    for i in range(1, G+1):
        GaugeCoordsTup = GaugeCoords(float(i)/(G+1))
        LonTuple.append(GaugeCoordsTup[0])
        LatTuple.append(GaugeCoordsTup[1])

    return tuple([tuple(LonTuple), tuple(LatTuple)])


def error_variance(A, Q, L, a, b):
    ''' variance of the attenuation measurement error.
    Formula (7) from paper.
    A - attenuation
    Q - quantizatin
    L - length in KM
    a, b - ITU power law parameters
    '''
    if A < 0.001:    # no rain induced attenuation
        A = 0.001

    return ((Q**2)/12.0) * (1/(a*L)) * (1.0/b) * (A**(2.0*(1.0-b)/b))


def calc_grid_weights(D, ROI, method, p_par, cml_lengths):
    ''' calculate shepard IDW coefficients/weights (for mapping)
    with radius of influence ROI. Formula (22) from paper. '''
    w = np.zeros(D.shape)
    D[D < 0.001] = 0.001
    if method==0:
        # restrain_term = 2.0*float(np.logical_not(cml_lengths>0)) + \
        #     float(cml_lengths>0)-((D[D < ROI])/(D[D < ROI]+(cml_lengths)))**p_par
        restrain_term = float(not np.any(cml_lengths)) + \
            1-((D[D < ROI])/(D[D < ROI]+(cml_lengths[D < ROI])))**p_par
        w[D < ROI] = \
            ((ROI/D[D < ROI] - 1.0)**p_par) * restrain_term
        w[w < 0.0000000000001] = 0.0
        # print(np.nanmean(restrain_term))
    else:
        w[D < ROI] = \
            (ROI**p_par - D[D < ROI]**p_par)/(ROI**p_par + D[D < ROI]**p_par)
        w[w < 0.0000000000001] = 0.0
    return w

class IdwIterative():
    def __init__(self, df_for_dist, xgrid, ygrid, ROI=0.0, max_iterations=1, 
                 tolerance=0.0, method=0, p_par=2.0, fixed_gmz_roi=None, restrain_w=False,
                 iteration0=False):
        '''ROI- Radius of Influence in meters. A parameter of IDW
        non-necessary variables:
        self.gauges_z_list
        
        '''
        self.df_for_dist = df_for_dist
        self.ROI = ROI
        self.ROI_gmz = ROI
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method # weighting method (0-shepard, 1-cressman)
        self.p_par = p_par # power parameter of IDW
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.ROI_prev = None
        self.fixed_gmz_roi = fixed_gmz_roi
        self.restrain_w = restrain_w
        self.iteration0 = iteration0

        self.df_for_dist['num_of_gauges'] = self.df_for_dist['x'].apply(len)
        # True for equidistant gauge distribution 
        self.ed_bool = self.df_for_dist.num_of_gauges.is_unique 
        if self.iteration0:
            print('Calculating weights for Iteration Zero')
            self._calc_iteration0()

        self._calc_all_weights()
        print('Calculation of weights finished')
                
    def __call__(self, df, quantization, interpolate=True,
                 ROI_for_interp=None):
        ''' Calculates the rain rate at every virtual gauge of every cml
        and uses the results to create a rain map with IDW interpolation.
        df is a pandas DataFrame that should include the following columns:
        x, y - (x,y) location of virtual gauges
        z - initial z value at each virtual gauge
        xa, ya, xb, yb - (x,y) location of the cml's sites
        L - length of each cml (in KM)
        a, b - ITU power law parameters
        A - attenuation due to rain only
        num_of_gauges - int. Number of virtual gauges
        '''
        if ROI_for_interp:
            self.ROI = ROI_for_interp
            
        # Check whether the same links are available for weights calculations
        if df.sort_values('Link_num')['Link_num'].equals(
            self.df_for_dist.sort_values('Link_num')['Link_num']
            ):
            print('Reusing precalculated weights')
            if ROI_for_interp:
                if self.ROI == self.ROI_prev:
                    print('Reusing precalculated weights')
                else:
                    print('Recalculating weights because' +\
                          ' CML IDs in input DataFrame did change')
                    self.df_for_dist = df
                    self._calc_all_weights()    
        else:
            print('Recalculating weights because' +\
                  ' CML IDs in input DataFrame did change')
            self.df_for_dist = df
            self._calc_all_weights()

        if ROI_for_interp:
            self.ROI_prev = ROI_for_interp
            
        # each vector contains data for all the gauges of all the links
        self.df = df
        self.Q = quantization
        self.df['num_of_gauges'] = self.df['x'].apply(len)

        # the grid points and Z values at each point (initialized to 0)
        self.Z = np.zeros((self.xgrid.shape[0], self.xgrid.shape[1]))

        # create vectors with x,y,z values of all the virtual gauges
        # self.gauges_z = np.zeros(self.use_gauges.shape)
        self.gauges_z = np.empty(self.use_gauges.shape) 
        self.gauges_z[:] = np.nan

        for cml_index, cml in self.df.iterrows():
            gauges = cml['num_of_gauges']
            self.gauges_z[cml_index, :gauges] = cml['z']

        self.gauges_z_prev = self.gauges_z.copy()
        
        # calculate the radius of influence
        if self.ROI == 0.0:
            self.calc_ROI()

        # run multiple iterations to find the virtual gauge values
        self.dz_vec = []
        dz = np.inf  # change in virtual gauge values
        self.gauges_z_list = []
        ## run iter 0 using the weights calculated in __init__
        if self.iteration0:
            self.df['variance'] = 0.0 # setting the variance to zero for iter zero
            print('Processing iteration Zero')
            self.calc_cmls_from_other_cmls(
                prev_gs=self.gauges_z_prev[:,0],
                use_iteration0_weights=True
            )
            self.gauges_z_prev[:, :] = self.gauges_z
            self.gauges_z_list.append(self.gauges_z)
            print('Processing iteration Zero done')
            self.dz_vec = [] 
            dz = np.inf
        # calculate the measurement error variance for each cml
        # these variance values do not change during each iteration
        self.df['variance'] = df.apply(lambda cml: error_variance(cml['A'],
                                                                  self.Q,
                                                                  cml['L'],
                                                                  cml['a'],
                                                                  cml['b']),
                                       axis=1) # CHECK

        for i in range(self.max_iterations):
            print('Running iteration %s' %i)

            # perform a single iteration on all the cmls
            self.calc_cmls_from_other_cmls(prev_gs=self.gauges_z_prev)

            if self.tolerance > 0.0:
                if self.ed_bool:
                    gauges_z_no_nans = self.gauges_z[~np.isnan(self.gauges_z)]
                    gauges_z_prev_no_nans = self.gauges_z_prev[~np.isnan(self.gauges_z_prev)]
                else:
                    gauges_z_no_nans = self.gauges_z
                    gauges_z_prev_no_nans = self.gauges_z_prev
                diff_norm = np.linalg.norm(gauges_z_no_nans - gauges_z_prev_no_nans,
                                           axis=None)
                prev_norm = np.linalg.norm(gauges_z_prev_no_nans, axis=None)
                dz = float(diff_norm)/(prev_norm + 1e-10)
                self.dz_vec.append(dz)
                print(prev_norm)
#                print('Norm change in Z values: {}'.format(dz))
                if dz <= self.tolerance:
                    break

            # update the previous z values for the next iteration
            self.gauges_z_prev[:, :] = self.gauges_z
            self.gauges_z_list.append(self.gauges_z)

        # calculate value at each grid point using the data from the cmls
        if interpolate == True:
            print('Interpolating\n')
            self.calc_grid_from_cmls()

        print('Processing finished.\n')
        return self.Z

    def calc_ROI(self):
        ''' Calculate the radius of influence by measuring the largest
        distance between any two cmls. '''

        # calculate the coordinates of each cml's center point
        cml_x_middle = self.df['x'].apply(lambda x: np.mean(x)).values
        cml_y_middle = self.df['y'].apply(lambda y: np.mean(y)).values

        cml_coords_middle = np.concatenate((cml_x_middle.reshape(-1, 1),
                                            cml_y_middle.reshape(-1, 1)),
                                           axis=1)
        dests = cml_coords_middle

        # calculate the distance between every pair of cmls
        subts = cml_coords_middle[:, None, :] - dests
        cml_distances = np.sqrt(np.einsum('ijk,ijk->ij', subts, subts))

        # use the max distance between any two cmls as the ROI
        self.ROI = cml_distances.max(axis=None)

    def calc_cmls_from_other_cmls(self, prev_gs ,use_iteration0_weights=False):
        ''' Isolate one cml at a time and compute the influence of other cmls
        in its radius of influence to calculate the rain rate at each of
        the cml's virtual gauges. '''
        if use_iteration0_weights:
            cmls_vg_with_neighbors = self.it0_cmls_vg_with_neighbors_list
            cml_vg_neighbors = self.it0_cml_vg_roi_list_all
            vgs_idx_neighbors = self.it0_cml_vg_roi_single_i_list_all
            weights_vg = self.it0_weights_vg_list
        else:
            cmls_vg_with_neighbors = self.cmls_vg_with_neighbors_list
            cml_vg_neighbors = self.cml_vg_roi_list_all
            vgs_idx_neighbors = self.cml_vg_roi_single_i_list_all
            weights_vg = self.weights_vg_list
        # compute the rain rate at each gauge of a cml (with index cml_i)
        # using the rain rate of virtual gauges from all the OTHER cmls
        try:
            cml_prev = cmls_vg_with_neighbors[0][0]
        except:
            raise Exception("No cmls with neighbors. Please check the units of " +\
                 "your Radius of Influence.")
        cml_num_of_gauges = self.df.iloc[cml_prev]['num_of_gauges']
        cml_new_z = np.zeros((cml_num_of_gauges,))
        last_i = len(cmls_vg_with_neighbors)-1
        for i, cml_vg in enumerate(cmls_vg_with_neighbors):
            if cml_vg[0] != cml_prev:
                self.gauges_z[cml_prev, :cml_num_of_gauges] = \
                    self._la_grange_mult(
                        K=cml_num_of_gauges,
                        R = self.df.iloc[cml_prev]['R'],
                        b = self.df.iloc[cml_prev]['b'],
                        cml_vg=cml_vg,
                        cml_new_z=cml_new_z
                    )
                cml_num_of_gauges = self.df.iloc[cml_vg[0]]['num_of_gauges']
                cml_new_z = np.zeros((cml_num_of_gauges,))
                cml_prev = cml_vg[0]

            # variances and number of gauges of cmls in ROI
            neighbor_cmls = []
            for _, k in enumerate(cml_vg_neighbors[i]):
                neighbor_cmls.append(k[0])
#                gauges_in_ROI.append(k[1])
            neighbor_cmls = list(set(neighbor_cmls))
            variances = self.df['variance'][neighbor_cmls].values

            # Initialize a covariance matrix to zero
            M = len(cml_vg_neighbors[i])  # total number of gauges in ROI
            cov = np.zeros((M, M))

            if use_iteration0_weights is False:
                neighbor_cmls_count_temp = [q[0] for q in cml_vg_neighbors[i]]
                count = Counter(neighbor_cmls_count_temp)
                dictionary_items = count.items()
                d_count = dict(sorted(dictionary_items))
                
                # add measurement quantization error to the cov matrix
                stop = 0
                for ic, c in enumerate(d_count):
                    start = stop
                    stop = start + d_count[c]
                    cov[start:stop, start:stop] = variances[ic]
            # add IDW weights to covariance matrix
            z = 1.0
            W = z*np.diagflat(1.0/weights_vg[i])  # 1/weights on diagonal
            cov = cov + W

            # compute inverse of covariance matrix
            cov_inv = np.linalg.inv(cov)
            cov_inv_sum = cov_inv.sum(axis=None)  # sum of all elements
            cov_inv_col_sum = cov_inv.sum(axis=0)
            prev_theta = np.take(prev_gs,
                vgs_idx_neighbors[i]
            ) # self.gauges_z_prev
            nominator = (cov_inv_col_sum * prev_theta).sum()
            cml_new_z[cml_vg[1]] = nominator / cov_inv_sum
            
            if i == last_i:
                self.gauges_z[cml_vg[0], :cml_num_of_gauges] = self._la_grange_mult(
                    K=cml_num_of_gauges,
                    R=self.df.iloc[cml_vg[0]]['R'],
                    b=self.df.iloc[cml_vg[0]]['b'],
                    cml_vg=cml_vg,
                    cml_new_z=cml_new_z
                )
    def _la_grange_mult(self,K,R,b,cml_vg,cml_new_z):
        '''Apply formula (20) from paper to compute the rain rate vector'''
        theta = cml_new_z**b
        r = (R**b - (1.0/K)*np.sum(theta) + theta)
        r[r < 0.0] = 0.0
        ## distribute negative rs to other vrgs
        # if np.sum(r[r < 0.0]) != 0:  
        #     print('NEGATIVE')
        #     rr = r
        #     ti = np.where(r >= 0.0)[0]
        #     fi = np.where(r < 0.0)[0]
        #     rain_for_dist = sum(rr[fi]) * -1
        #     rain_for_dist_val = rain_for_dist / len(ti)
        #     r[ti] += rain_for_dist_val
        #     r[fi] = 0.0 # set negative rain rates to 0
        return r**(1.0/b)

    def calc_grid_from_cmls(self):
        ''' calculate the z values of each grid point using z values of the
        virtual gauges '''
        Z_flat = self.Z.flatten()
        # use the cmls to caluculate the rain rate at each (x,y) grid point
        for i, p_i in enumerate(self.pixels_with_neighbors_list):
            gauges_i = np.take(self.gauges_z, 
                               self.idx_pxl_single_i_list_all[i])
            Z_flat[p_i] = (self.weights_pxl_list_all[i] * gauges_i).sum()/\
            self.sum_of_weights[i]
#            else:
        Z_flat[Z_flat <= 0] = np.nan
        self.Z = Z_flat.reshape((self.xgrid.shape[0], self.xgrid.shape[1]))
        

    def _calc_all_weights(self):
        # gmz calcs:
        self.num_cmls = self.df_for_dist.shape[0]
        # compute the number of virtual gauges for each cml
        self.max_num_of_gauges = self.df_for_dist['num_of_gauges'].max()
        self.use_gauges = np.zeros((self.num_cmls, self.max_num_of_gauges),
                                   dtype=bool)

        self.cml_lengths = np.tile(self.df_for_dist['L'].values,
            (self.max_num_of_gauges,1)).transpose()*float(self.restrain_w)
        self.gauges_x = np.empty(self.use_gauges.shape)
        self.gauges_x[:] = np.nan 
        self.gauges_y = np.empty(self.use_gauges.shape)
        self.gauges_y[:] = np.nan 
        
        for cml_index, cml in self.df_for_dist.iterrows():
            gauges = cml['num_of_gauges']
            self.use_gauges[cml_index, :gauges] = True
            self.gauges_x[cml_index, :gauges] = cml['x'] 
            self.gauges_y[cml_index, :gauges] = cml['y']
            # self.cml_lengths[cml_index, :gauges] = self.df_for_dist['L'].values#*1e3

        self.cmls_vg_with_neighbors_list = []
        self.cml_vg_roi_list_all = []
        self.cml_vg_roi_single_i_list_all = []
        self.weights_vg_list = []
        for cml_i, cml in self.df_for_dist.iterrows():    # loop over cmls
            cml_gx = cml['x']   # x position of virtual gauges
            cml_gy = cml['y']   # y position of virtual gauges

            # initial new virtual gauge rain vector for current cml
            cml_num_of_gauges = cml['num_of_gauges']
            # cml_length = float(self.restrain_w) * cml['L']#*1e3 
            # loop over current cml's virtual gauges
            for gauge_i in range(cml_num_of_gauges):
                cml_vg_roi_list = []
                cml_vg_roi_single_i_list = []
                gx = cml_gx[gauge_i]  # x position of current gauge
                gy = cml_gy[gauge_i]  # y position of currnet gauge

                # calculate distance of current gauge from all gauges
                distances = np.sqrt((self.gauges_x - gx)**2.0 +
                                    (self.gauges_y - gy)**2.0)

                ## calculate IDW weights
                if self.fixed_gmz_roi:
                    weights = calc_grid_weights(distances,
                                                self.fixed_gmz_roi,
                                                self.method,
                                                self.p_par,
                                                cml_lengths=self.cml_lengths)

                else:
                    weights = calc_grid_weights(distances, 
                                                self.ROI, 
                                                self.method, 
                                                self.p_par,
                                                cml_lengths=self.cml_lengths)
                weights = weights * self.use_gauges
                weights[cml_i, :] = 0.0  # remove weights for current cml
                weights[weights < weights.max()/1000.0] = 0.0

                # find the indices of cmls in the current cml's ROI
                cmls_in_ROI = (weights.sum(axis=1) > 0.0)
                gauges_in_ROI = (weights > 0.0)

                # exclude current cml from all further calculations
                cmls_in_ROI[cml_i] = False
                gauges_in_ROI[cml_i, :] = False

                # select the indices of cml gauges (only in the ROI)
                select_gauges = gauges_in_ROI * self.use_gauges
                if select_gauges.sum() != 0:
                    self.cmls_vg_with_neighbors_list.append((cml_i, gauge_i))
                    for row in range(len(select_gauges[:,0])):
                        if select_gauges[row,:].sum() != 0:
                            for col in range(self.max_num_of_gauges):
                                if select_gauges[row,col].sum() != 0: # CAN REMOVE SUM
                                    cml_vg_roi_list.append((row,col))
                                    cml_vg_roi_single_i_list.\
                                    append(row*self.max_num_of_gauges + col)
                # add IDW weights to covariance matrix
#                z = 0.5
                self.cml_vg_roi_single_i_list_all.append(
                    cml_vg_roi_single_i_list
                    )
                self.cml_vg_roi_single_i_list_all = \
                list(filter(None, self.cml_vg_roi_single_i_list_all))
                self.cml_vg_roi_list_all.append(cml_vg_roi_list)
                self.cml_vg_roi_list_all = list(
                    filter(None, 
                           self.cml_vg_roi_list_all)
                    )
                weights_vector = weights[select_gauges].flatten()
                if len(weights_vector) != 0:
                    weights_vector /= weights_vector.sum(axis=None)  # Normalize
                    self.weights_vg_list.append(weights_vector) # List of vgs for the cov matrix
        self._calc_cml_to_point_weights(
            x_target_vec=self.xgrid.flatten(),
            y_target_vec=self.ygrid.flatten(),
            x_source_arr=self.gauges_x,
            y_source_arr=self.gauges_y
            )

    def _calc_cml_to_point_weights(self,x_target_vec,y_target_vec,x_source_arr,y_source_arr): 
        '''
        x_target_vec,y_target_vec- 1D vectors of x,y locations
        x_source_arr,y_source_arr- arrays of x,y locations
        '''          
        # for idw pixels calcs
        self.pixels_with_neighbors_list = []
        self.idx_weights_pxl_list_all = []
        self.weights_pxl_list_all = []
        self.idx_pxl_single_i_list_all = []
        for p_i in range(len(x_target_vec)):
            px = x_target_vec[p_i]
            py = y_target_vec[p_i]
            # perform basic IDW
            idx_weights_pxl_list = []
            weights_pxl_list = []
            idx_pxl_single_i_list = []
            dist_pxl = np.sqrt((x_source_arr-px)**2 + (y_source_arr-py)**2)
            if self.fixed_gmz_roi:
                weights_pxl = calc_grid_weights(dist_pxl, 
                                                self.fixed_gmz_roi, 
                                                self.method, 
                                                self.p_par,
                                                cml_lengths=self.cml_lengths)
            else:
                weights_pxl = calc_grid_weights(dist_pxl, 
                                                self.ROI, 
                                                self.method, 
                                                self.p_par,
                                                cml_lengths=self.cml_lengths)
            weights_pxl[np.bitwise_not(self.use_gauges)] = 0.0 # Check!
            if weights_pxl.sum() != 0:
                    self.pixels_with_neighbors_list.append(p_i)
                    for row in range(len(weights_pxl[:,0])):
                        if weights_pxl[row,:].sum() != 0:
                            for col in range(self.max_num_of_gauges):
                                if weights_pxl[row,col].sum() != 0: # CAN REMOVE SUM
                                    idx_weights_pxl_list.append((row,col))
                                    weights_pxl_list.append(
                                        weights_pxl[row,col]
                                        )
                                    idx_pxl_single_i_list.\
                                    append(row*self.max_num_of_gauges + col)
            self.idx_weights_pxl_list_all.append(idx_weights_pxl_list)
            self.idx_weights_pxl_list_all = list(
                filter(None, self.idx_weights_pxl_list_all)
                )
            self.weights_pxl_list_all.append(weights_pxl_list)
            self.weights_pxl_list_all = list(
                filter(None,self.weights_pxl_list_all)
                )
            self.idx_pxl_single_i_list_all.append(idx_pxl_single_i_list)
            self.idx_pxl_single_i_list_all = \
                list(filter(None, self.idx_pxl_single_i_list_all))
            
        self.sum_of_weights = []
        for i, w_vec_pxl in enumerate(self.weights_pxl_list_all):
            w_vec_pxl = np.array(w_vec_pxl)
            self.sum_of_weights.append(w_vec_pxl.sum(axis=None))

    def _calc_iteration0(self):
        x_mids = []
        y_mids = []

        # split each cml into several virtual rain gauges
        for _, cml in self.df_for_dist.iterrows():
            x, y = get_gauges_lon_lat(cml['xa'], cml['ya'],
                                        cml['xb'], cml['yb'],
                                        G=1)
            x_mids.append(x)
            y_mids.append(y)

        ## add x, y locations of the virtual rain gauges of each cml
        self.df_for_dist['x_mid'] = x_mids
        self.df_for_dist['y_mid'] = y_mids

        ## gmz calcs:
        num_cmls = self.df_for_dist.shape[0]
        ## compute the number of virtual gauges for each cml
        it0_use_gauges = np.ones((num_cmls, 1),
                                   dtype=bool)
        gauges_x_it0 = np.empty(it0_use_gauges.shape)
        gauges_x_it0[:] = np.nan 
        gauges_y_it0 = np.empty(it0_use_gauges.shape)
        gauges_y_it0[:] = np.nan 
        
        self.it0_cmls_vg_with_neighbors_list = []
        self.it0_cml_vg_roi_list_all = []
        self.it0_cml_vg_roi_single_i_list_all = []
        self.it0_weights_vg_list = []

        for cml_index, cml in self.df_for_dist.iterrows():
            gauges_x_it0[cml_index, :1] = cml['x_mid'] 
            gauges_y_it0[cml_index, :1] = cml['y_mid']

        for cml_i, cml in self.df_for_dist.iterrows():    # loop over cmls
            cml_gx = cml['x']   # x position of virtual gauges
            cml_gy = cml['y']   # y position of virtual gauges
            # initial new virtual gauge rain vector for current cml
            cml_num_of_gauges = cml['num_of_gauges']
            # cml_length = float(self.restrain_w) * cml['L']#*1e3 
            # loop over current cml's virtual gauges
            for gauge_i in range(cml_num_of_gauges):
                cml_vg_roi_list = []
                cml_vg_roi_single_i_list = []
                gx = cml_gx[gauge_i]  # x position of current gauge
                gy = cml_gy[gauge_i]  # y position of currnet gauge

                # calculate distance of current gauge from all gauges
                distances = np.sqrt((gauges_x_it0 - gx)**2.0 +
                                    (gauges_y_it0 - gy)**2.0)
                ## calculate IDW weights
                weights = calc_grid_weights(distances, 
                                            self.ROI, 
                                            self.method, 
                                            self.p_par,
                                            cml_lengths=np.zeros(distances.shape))
                # weights = weights * it0_use_gauges
                weights[cml_i, :] = 0.0  # remove weights for current cml
                weights[weights < weights.max()/1000.0] = 0.0

                # find the indices of cmls in the current cml's ROI
                cmls_in_ROI = (weights.sum(axis=1) > 0.0) # shape (cmls,)
                gauges_in_ROI = (weights > 0.0) # shape (cmls,1)

                # exclude current cml from all further calculations
                cmls_in_ROI[cml_i] = False
                gauges_in_ROI[cml_i, :] = False
                # select the indices of cml gauges (only in the ROI)
                select_gauges = gauges_in_ROI# * it0_use_gauges
                if select_gauges.sum() != 0:
                    self.it0_cmls_vg_with_neighbors_list.append((cml_i, gauge_i))
                    for row in range(len(select_gauges[:,0])):
                        if select_gauges[row,:].sum() != 0:
                            if select_gauges[row,0].sum() != 0: # CAN REMOVE SUM
                                cml_vg_roi_list.append((row,0))
                                cml_vg_roi_single_i_list.append(row)
                # add IDW weights to covariance matrix
#                z = 0.5
                self.it0_cml_vg_roi_single_i_list_all.append(
                    cml_vg_roi_single_i_list
                    )
                self.it0_cml_vg_roi_single_i_list_all = \
                list(filter(None, self.it0_cml_vg_roi_single_i_list_all))
                self.it0_cml_vg_roi_list_all.append(cml_vg_roi_list)
                self.it0_cml_vg_roi_list_all = list(
                    filter(None, 
                           self.it0_cml_vg_roi_list_all)
                    )
                weights_vector = weights[select_gauges].flatten()
                if len(weights_vector) != 0:
                    weights_vector /= weights_vector.sum(axis=None)  # Normalize
                    self.it0_weights_vg_list.append(weights_vector) # List of vgs for the cov matrix