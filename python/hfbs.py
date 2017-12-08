#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import time
import datetime
import argparse
from multiprocessing import Process, Queue

# local imports
import utils

#flags
VERBOSE = True

# constants
EPS = np.finfo(float).eps
np.set_printoptions(threshold=np.nan)
BILATERAL_SIGMA_LUMA = 16
BILATERAL_SIGMA_SPATIAL = 8
LOSS_SMOOTH_MULT = 8
A_VERT_DIAG_MIN = 1e-3
NUM_PCG_ITERS = 30
NUM_NEIGHBORS = 6

reference = '../data/depth_superres/reference.png'
target = '../data/depth_superres/target.png'
confidence = '../data/depth_superres/confidence.png'

def prepare_flow(reference, flow_tuple, confidence):
    reference_image = np.array(plt.imread(reference, format='png'), dtype=np.float32)*256
    flow = np.array(plt.imread(flow_tuple[0], format='png'), dtype=np.float32)*65536
    flow = np.subtract(flow, 2**15)
    flow = np.divide(flow, 256)

    weight = np.array(plt.imread(confidence, format='png'), dtype=np.float32)*65536
    weight = np.divide(weight, 65536)

    if VERBOSE:
        print(">>> preparing flow data")
    sz = [reference_image.shape[0], reference_image.shape[1]]

    I_x = np.tile(np.floor(np.divide(np.arange(sz[1]), BILATERAL_SIGMA_SPATIAL)), (sz[0],1))
    I_y = np.tile( np.floor(np.divide(np.arange(sz[0]), BILATERAL_SIGMA_SPATIAL)).reshape(1,-1).T, (1,sz[1]) )
    I_luma = np.floor_divide(utils.rgb2gray(reference_image), float(BILATERAL_SIGMA_LUMA))

    X = np.concatenate((I_x[:,:,None],I_y[:,:,None],I_luma[:,:,None]),axis=2).reshape((-1,3),order='F')
    W0 = np.ravel(weight.T)
    X0 = np.reshape(flow,[-1,1],order='F')
    return X, W0, X0, flow.shape

def make_neighbors(grid_size):
    ii,jj,kk = np.mgrid[0:grid_size[0],0:grid_size[1],0:grid_size[2]]

    idxs0 = np.array([], dtype=np.int64).reshape(0,2)
    idxs = np.array([], dtype=np.int64).reshape(0,2)

    if VERBOSE:
        print(">>> neighbors")

    for diff in [-1,1]:

        ii_ = ii + diff
        jj_ = jj + diff
        kk_ = kk + diff

        # first ii
        idx = np.array([ii[(0 <= ii_) & (ii_ < grid_size[0])],
            jj[(0 <= ii_) & (ii_ < grid_size[0])],
            kk[(0 <= ii_) & (ii_ < grid_size[0])]]).T
        idx2ravel = np.array([np.ravel_multi_index(item,dims=grid_size,order='F') for item in idx])

        idx_ = np.array([ii_[(0 <= ii_) & (ii_ < grid_size[0])],
            jj[(0 <= ii_) & (ii_ < grid_size[0])],
            kk[(0 <= ii_) & (ii_ < grid_size[0])]]).T
        idx_2ravel = np.array([np.ravel_multi_index(item,dims=grid_size,order='F') for item in idx_])

        idxs = np.vstack( [idxs, np.array([idx2ravel,idx_2ravel]).T ] )

        # then jj
        idx = np.array([ii[(0 <= jj_) & (jj_ < grid_size[1])],
            jj[(0 <= jj_) & (jj_ < grid_size[1])],
            kk[(0 <= jj_) & (jj_ < grid_size[1])]]).T
        idx2ravel = np.array([np.ravel_multi_index(item,dims=grid_size,order='F') for item in idx])

        idx_ = np.array([ii[(0 <= jj_) & (jj_ < grid_size[1])],
            jj_[(0 <= jj_) & (jj_ < grid_size[1])],
            kk[(0 <= jj_) & (jj_ < grid_size[1])]]).T
        idx_2ravel = np.array([np.ravel_multi_index(item,dims=grid_size,order='F') for item in idx_])

        idxs = np.vstack( [idxs, np.array([idx2ravel,idx_2ravel]).T ] )

        # then kk
        idx = np.array([ii[(0 <= kk_) & (kk_ < grid_size[2])],
            jj[(0 <= kk_) & (kk_ < grid_size[2])],
            kk[(0 <= kk_) & (kk_ < grid_size[2])]]).T
        idx2ravel = np.array([np.ravel_multi_index(item,dims=grid_size,order='F') for item in idx])

        idx_ = np.array([ii[(0 <= kk_) & (kk_ < grid_size[2])],
            jj[(0 <= kk_) & (kk_ < grid_size[2])],
            kk_[(0 <= kk_) & (kk_ < grid_size[2])]]).T
        idx_2ravel = np.array([np.ravel_multi_index(item,dims=grid_size,order='F') for item in idx_])

        idxs = np.vstack( [idxs, np.array([idx2ravel,idx_2ravel]).T ] )
    return idxs


def bistochastize(splat_mat, grid_size, diffuse3_mat, W0, Xshape):
    splat_sum = np.reshape( splat_mat * np.ones(Xshape), grid_size,order='F').astype('float32')
    splat_norm = np.ones(splat_sum.shape).astype('float32')

    for i_norm in range(20):
        diffuse_splat_norm = np.reshape(diffuse3_mat * np.reshape(splat_norm, -1, 1), splat_norm.shape,order='F')
        blurred_splat_norm = 2 * splat_norm + diffuse_splat_norm
        denom = np.maximum(EPS, blurred_splat_norm)
        splat_norm = np.sqrt(splat_norm * (splat_sum / denom))


    A_diag = np.reshape((splat_mat * W0), grid_size,order='F') + LOSS_SMOOTH_MULT * (splat_sum - 2 * np.square(splat_norm))
    return splat_norm, A_diag



def dense_solve_jacobi(X, W0, X0):
    if VERBOSE:
        print(">>> starting bilateral grid optimization")

    grid_size = (np.amax(X,axis=0) + np.ones(3)).astype('int64')
    basis = np.insert(np.cumprod(grid_size)[0:-1],0,1)

    grid_num_vertices = np.prod(grid_size)
    grid_splat_indices = np.dot(X,basis)

    # this is where the info to build the sparse matrix is:
    # http://stackoverflow.com/questions/7760937/issue-converting-matlab-sparse-code-to-numpy-scipy-with-csc-matrix
    splat_ones = np.ones(grid_splat_indices.shape[0])
    ij = ( grid_splat_indices-1, np.arange(0, grid_splat_indices.shape[0]) )
    splat_mat_shape = ( np.prod(grid_size), X.shape[0] )

    splat_mat = csr_matrix( (splat_ones, ij), shape=splat_mat_shape )

    start_new_neighbors = time.time()
    idxs = make_neighbors(grid_size)
    if VERBOSE:
        print("time neighbors: ", time.time() - start_new_neighbors)

    d_shape = np.prod(grid_size)
    diffuse_s = np.tile(1,idxs.shape[0])
    diffuse3_mat = csr_matrix( (diffuse_s, (idxs[:,0],idxs[:,1])), shape=(d_shape,d_shape) )

    if VERBOSE:
        print(">>> bistochastization")

    start_bistoch = time.time()
    splat_norm, A_diag = bistochastize(splat_mat, grid_size, diffuse3_mat, W0, X.shape[0])
    if VERBOSE:
        print("time bistoch: ", time.time() - start_bistoch)

    XW0 = (W0*X0.T).T.astype('float32')
    b_mat = np.reshape((splat_mat * XW0), grid_size,order='F').astype('float32')

    #% Construct A, b, and c, which define our optimization problem
    c = 0.5 * sum(XW0 * X0,1);

    # initial solution
    inv_a = 1/np.maximum(A_VERT_DIAG_MIN, A_diag)
    #inv_a = np.divide(1,np.maximum(A_VERT_DIAG_MIN, A_diag)).astype('float32')

    final_result_cpu = []

    # for each channel in the flow
    if VERBOSE:
        print(">>> starting flow!")

    b_c = b_mat[:,:,:].astype('float32')
    Y_c = np.multiply(b_c, inv_a)
    Y_c_cpu =  np.multiply(b_c, inv_a)
    Ysn = np.multiply(Y_c, splat_norm)

    start_vector = time.time()
    for iter in range(NUM_PCG_ITERS):

        Ysn = np.multiply(Y_c, splat_norm) # flow 1
        Ysn_reshaped = np.reshape(Ysn,-1,1)
        Ysn_r2 = np.reshape((diffuse3_mat * Ysn_reshaped), Ysn.shape, order='F')
        temp_Y = (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 ))

        Y_c_cpu = np.multiply((b_c - temp_Y) , inv_a)
        Y_c = Y_c_cpu

        total_time_vector = time.time() - start_vector #end timing
        Y_c_cpu = Y_c

    final_result_cpu.append(Y_c_cpu)

    if VERBOSE:
        print(">>> completed, total time:\t", time.time() - start_vector)

    final_solution_cpu = final_result_cpu

    sliced_solution_cpu = (splat_mat.T) * (final_solution_cpu[0].reshape(grid_num_vertices,-1,order='F'))

    return sliced_solution_cpu # final result is an array with improved flows in both directions


def main():
    input_X, input_W0, input_X0, flow_shape = prepare_flow(reference,(target,), confidence)

    res_cpu = dense_solve_jacobi(input_X, input_W0, input_X0)
    res_cpu = res_cpu.reshape(flow_shape[0], flow_shape[1], -1,order='F')

    ref = np.array(plt.imread(reference, format='png'), dtype=np.float32)
    flow = np.array(plt.imread(target, format='png'), dtype=np.float32)*65536
    flow = np.subtract(flow, 2**15)
    flow = np.divide(flow, 256)
    conf = np.array(plt.imread(confidence, format='png'), dtype=np.float32)*256*256
    conf = np.divide(conf, 65536)

    plt.figure(figsize=(5,10))

    plt.subplot(4,1,1)
    plt.imshow(ref)
    plt.title('reference')
    plt.colorbar()

    plt.subplot(4,1,2)
    plt.imshow(flow)
    plt.title('input flow')
    plt.colorbar()

    plt.subplot(4,1,3)
    plt.imshow(conf)
    plt.title('confidence')
    plt.colorbar()

    plt.subplot(4,1,4)
    plt.imshow(res_cpu[:,:,0])
    plt.title('output flow')
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    main()
