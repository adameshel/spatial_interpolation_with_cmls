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
        L = cml['L']
        if num_gauges is None:
#            import pdb; pdb.set_trace()
            num_gauges_along_cml = int(np.ceil(L / float(gauge_length)))
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


def calc_grid_weights(D, ROI, method, p_par):
    ''' calculate shepard IDW coefficients/weights (for mapping)
    with radius of influence ROI. Formula (22) from paper. '''
    w = np.zeros(D.shape)
    D[D < 0.001] = 0.001
    if method==0:
        w[D < ROI] = (ROI/D[D < ROI] - 1.0)**p_par
        w[w < 0.0000000000001] = 0.0
    else:
        w[D < ROI] = \
            (ROI**p_par - D[D < ROI]**p_par)/(ROI**p_par + D[D < ROI]**p_par)
        w[w < 0.0000000000001] = 0.0
    return w

class IdwIterative():
    def __init__(self, df_for_dist, xgrid, ygrid, ROI=0.0, max_iterations=1, 
                 tolerance=0.0, method=0, p_par=2.0, fixed_gmz_roi=None):
        '''ROI- Radius of Influence in meters. A parameter of IDW'''
        self.ROI = ROI
        self.ROI_gmz = ROI
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method # weighting method (0-shepard, 1-cressman)
        self.p_par = p_par # power parameter of IDW
        self.df_for_dist = df_for_dist
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.ROI_prev = None
        self.fixed_gmz_roi = fixed_gmz_roi
        
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
        '''
        if ROI_for_interp:
            self.ROI = ROI_for_interp
            
        # Check whether the same links are available for weights calculations
        if df['Link_num'].equals(self.df_for_dist['Link_num']):
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
#        import pdb; pdb.set_trace()
        self.df['num_of_gauges'] = self.df['x'].apply(len)
        # the grid points and Z values at each point (initialized to 0)
        self.Z = np.zeros((self.xgrid.shape[0], self.xgrid.shape[1]))

        # create vectors with x,y,z values of all the virtual gauges
        self.gauges_z = np.zeros(self.use_gauges.shape)

        for cml_index, cml in self.df.iterrows():
            gauges = cml['num_of_gauges']
            self.gauges_z[cml_index, :gauges] = cml['z']

        self.gauges_z_prev = self.gauges_z.copy()

        # calculate the measurement error variance for each cml
        # these variance values do not change during each iteration
        self.df['variance'] = df.apply(lambda cml: error_variance(cml['A'],
                                                                  self.Q,
                                                                  cml['L'],
                                                                  cml['a'],
                                                                  cml['b']),
                                       axis=1) # CHECK
        
        # calculate the radius of influence
        if self.ROI == 0.0:
            self.calc_ROI()

        # run multiple iterations to find the virtual gauge values
        self.dz_vec = []
        dz = np.inf  # change in virtual gauge values

        for i in range(self.max_iterations):
#            print('Running iteration {}'.format(i))

            # perform a single iteration on all the cmls
            self.calc_cmls_from_other_cmls()

            if self.tolerance > 0.0:
                diff_norm = np.linalg.norm(self.gauges_z - self.gauges_z_prev,
                                           axis=None)
                prev_norm = np.linalg.norm(self.gauges_z_prev, axis=None)
                dz = float(diff_norm)/(prev_norm + 1e-10)
                self.dz_vec.append(dz)
#                print('Norm change in Z values: {}'.format(dz))
                if dz <= self.tolerance:
                    break

            # update the previous z values for the next iteration
            self.gauges_z_prev[:, :] = self.gauges_z

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

    def calc_cmls_from_other_cmls(self):
        ''' Isolate one cml at a time and compute the influence of other cmls
        in its radius of influence to calculate the rain rate at each of
        the cml's virtual gauges. '''

        # compute the rain rate at each gauge of a cml (with index cml_i)
        # using the rain rate of virtual gauges from all the OTHER cmls
        cml_prev = self.cmls_vg_with_neighbors_list[0][0]
        cml_num_of_gauges = self.df.iloc[cml_prev]['num_of_gauges']
        cml_new_z = np.zeros((cml_num_of_gauges,))
        last_i = len(self.cmls_vg_with_neighbors_list)-1
#        import pdb; pdb.set_trace()
        for i, cml_vg in enumerate(self.cmls_vg_with_neighbors_list):
            # Apply formula (20) from paper to compute the rain rate vector
            if cml_vg[0] != cml_prev:
                K = cml_num_of_gauges
                R = self.df.iloc[cml_prev]['R']
                b = self.df.iloc[cml_prev]['b']
                theta = cml_new_z**b
                r = (R**b - (1.0/K)*np.sum(theta) + theta)
                # print(np.sum(r[r < 0.0]))
    #            import pdb; pdb.set_trace()
                r[r < 0.0] = 0.0  # set negative rain rates to 0
    
                # update the new z values at the cml's virtual gauges
                self.gauges_z[cml_prev, :cml_num_of_gauges] = r**(1.0/b)
                # initial new virtual gauge rain vector for current cml
                cml_num_of_gauges = self.df.iloc[cml_vg[0]]['num_of_gauges']
                cml_new_z = np.zeros((cml_num_of_gauges,))
                cml_prev = cml_vg[0]

            # variances and number of gauges of cmls in ROI
            neighbor_cmls = []
            for _, k in enumerate(self.cml_vg_roi_list_all[i]):
                neighbor_cmls.append(k[0])
#                gauges_in_ROI.append(k[1])
            neighbor_cmls = list(set(neighbor_cmls))
            variances = self.df['variance'][neighbor_cmls].values

            # Initialize a covariance matrix to zero
            M = len(self.cml_vg_roi_list_all[i])  # total number of gauges in ROI
            cov = np.zeros((M, M))

            neighbor_cmls_count_temp = [q[0] for q in self.cml_vg_roi_list_all[i]]
            count = Counter(neighbor_cmls_count_temp)
            dictionary_items = count.items()
            d_count = dict(sorted(dictionary_items))
            
            # add measurement quantization error to the cov matrix
            stop = 0
            for ic, c in enumerate(d_count):
            # for tt, sigma in enumerate(variances):
                start = stop
                stop = start + d_count[c]
                cov[start:stop, start:stop] = variances[ic]
            # add IDW weights to covariance matrix
            z = 1.0
            W = z*np.diagflat(1.0/self.weights_vg_list[i])  # 1/weights on diagonal
            cov = cov + W

            # compute inverse of covariance matrix
            cov_inv = np.linalg.inv(cov)
            cov_inv_sum = cov_inv.sum(axis=None)  # sum of all elements
            cov_inv_col_sum = cov_inv.sum(axis=0)
            prev_theta = np.take(self.gauges_z_prev, \
                                 self.cml_vg_roi_single_i_list_all[i])
            nominator = (cov_inv_col_sum * prev_theta).sum()
#            if i==2:
#                import pdb; pdb.set_trace()
            cml_new_z[cml_vg[1]] = nominator / cov_inv_sum
            
            # Apply formula (20) from paper to compute the rain rate vector
#            print(cml_i)
            if i == last_i:
                K = cml_num_of_gauges
                R = self.df.iloc[cml_vg[0]]['R']
                b = self.df.iloc[cml_vg[0]]['b']
                theta = cml_new_z**b
                r = (R**b - (1.0/K)*np.sum(theta) + theta)
                r[r < 0.0] = 0.0  # set negative rain rates to 0
                # update the new z values at the cml's virtual gauges
                self.gauges_z[cml_vg[0], :cml_num_of_gauges] = r**(1.0/b)

    def calc_grid_from_cmls(self):
        ''' calculate the z values of each grid point using z values of the
        virtual gauges '''
        Z_flat = self.Z.flatten()
        # use the cmls to caluculate the rain rate at each (x,y) grid point
        for i, p_i in enumerate(self.pixels_with_neighbors_list):
            gauges_i = np.take(self.gauges_z, 
                               self.idx_pxl_single_i_list_all[i])
#            import pdb; pdb.set_trace()
            Z_flat[p_i] = (self.weights_pxl_list_all[i] * gauges_i).sum()/\
            self.sum_of_weights[i]
#            else:
        Z_flat[Z_flat <= 0] = np.nan
        self.Z = Z_flat.reshape((self.xgrid.shape[0], self.xgrid.shape[1]))
        

    def _calc_all_weights(self):
        # gmz calcs:
        self.num_cmls = self.df_for_dist.shape[0]
        # compute the number of virtual gauges for each cml
        self.df_for_dist['num_of_gauges'] = self.df_for_dist['x'].apply(len)
        self.max_num_of_gauges = self.df_for_dist['num_of_gauges'].max()
        self.use_gauges = np.zeros((self.num_cmls, self.max_num_of_gauges),
                                   dtype=bool)
        self.gauges_x = np.zeros(self.use_gauges.shape) 
        self.gauges_y = np.zeros(self.use_gauges.shape)
            
        for cml_index, cml in self.df_for_dist.iterrows():
            gauges = cml['num_of_gauges']
            self.use_gauges[cml_index, :gauges] = True
            self.gauges_x[cml_index, :gauges] = cml['x'] 
            self.gauges_y[cml_index, :gauges] = cml['y']

        self.cmls_with_neighbors_list = []
        self.cmls_vg_with_neighbors_list = []
        self.cml_vg_roi_list_all = []
        self.cml_vg_roi_single_i_list_all = []
        self.weights_vg_list = []
        for cml_i, cml in self.df_for_dist.iterrows():    # loop over cmls
            cml_gx = cml['x']   # x position of virtual gauges
            cml_gy = cml['y']   # y position of virtual gauges

            # initial new virtual gauge rain vector for current cml
            cml_num_of_gauges = cml['num_of_gauges']
            
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
                                                self.p_par)

                else:
                    weights = calc_grid_weights(distances, 
                                                self.ROI, 
                                                self.method, 
                                                self.p_par)
                weights = weights * self.use_gauges
                weights[cml_i, :] = 0.0  # remove weights for current cml
                weights[weights < weights.max()/100.0] = 0.0

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
                    self.weights_vg_list.append(weights_vector)

                    
        # for idw pixels calcs
        self.pixels_with_neighbors_list = []
        self.idx_weights_pxl_list_all = []
        self.weights_pxl_list_all = []
        self.idx_pxl_single_i_list_all = []
        for p_i in range(len(self.xgrid.flatten())):
            px = self.xgrid.flatten()[p_i]
            py = self.ygrid.flatten()[p_i]
#            import pdb; pdb.set_trace()
            # perform basic IDW
            idx_weights_pxl_list = []
            weights_pxl_list = []
            idx_pxl_single_i_list = []
            dist_pxl = np.sqrt((self.gauges_x-px)**2 + (self.gauges_y-py)**2)
            if self.fixed_gmz_roi:
                weights_pxl = calc_grid_weights(dist_pxl, 
                                                self.fixed_gmz_roi, 
                                                self.method, 
                                                self.p_par)
            else:
                weights_pxl = calc_grid_weights(dist_pxl, 
                                                self.ROI, 
                                                self.method, 
                                                self.p_par)
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