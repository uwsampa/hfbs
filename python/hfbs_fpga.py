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
RUN_FPGA = False
VERBOSE = True
SMALL = True
FIXEDP = True
LOOP_OUTSIDE = False
NO_GARBAGE_ADDED = True
SAVE_FIG = False
DISPLAY_FIG = True
TWO_FLOW = False

# settings
EPS = np.finfo(float).eps
np.set_printoptions(threshold=np.nan)

# constants
BILATERAL_SIGMA_LUMA = 32
BILATERAL_SIGMA_SPATIAL = 32
LOSS_SMOOTH_MULT = 8
A_VERT_DIAG_MIN = 1e-3
NUM_PCG_ITERS = 100
NUM_NEIGHBORS = 6

if SMALL:
    reference = 'img/reference_sm.png'
    target = 'img/input0_sm.png'
    confidence = 'img/weight_sm.png'
else:
    reference = 'img/Motorcycle/reference.png'
    target = 'img/Motorcycle/input.png'
    confidence = 'img/Motorcycle/weight.png'

# FPGA-specific constants, yours may be different
c2h_str = "/dev/xdma/card0/c2h0"
h2c_str = "/dev/xdma/card0/h2c0"

def read_from_fpga(fpga_read_channel, num_bytes, my_queue):

    if not LOOP_OUTSIDE:
        print("reading (num_bytes-8)/5)+8", 8+(num_bytes-8)/5)
        data_readback = os.read(fpga_read_channel, ((num_bytes-8)/5)+8)
    else:
        if NO_GARBAGE_ADDED:
            print "reading 3*(num_bytes/10)", 3*(num_bytes/10)
            data_readback = os.read(fpga_read_channel, 3*(num_bytes/10))
        else:
            data_readback = os.read(fpga_read_channel, num_bytes)
    #data_readback = os.read(fpga_read_channel, 8)
    print "\tdone reading, putting"
    my_queue.put(data_readback)
    return

def invoke_fpga(im):

    result = ''

    c2h_fd = 0
    h2c_fd = 0
    new_im = ''.join([utils.to_bytestr(num) for num in im])
    try:
        # open up read and write channels
        h2c_fd = os.open(h2c_str, os.O_RDWR)
        c2h_fd = os.open(c2h_str, os.O_RDWR)

        bytes_left = im.size
        result_queue = Queue()

        # processes to handle r/w channels
        read_process = Process(target=read_from_fpga, args=(c2h_fd,len(new_im), result_queue))
        # write_process = Process(target=write_to_fpga, args=(h2c_fd,buffer(im, im.size - bytes_left)))

        # open up rx
        read_process.start()
        print "started read"
        time.sleep(.5) # wait for the read to start TODO do this in the correct way (maybe wait until poll != 0 )

        # open up tx, tx should trigger rx
        print "writing"
        if FIXEDP:
            bytes_written = os.write(h2c_fd, buffer(new_im, 0))
        else:
            bytes_written = os.write(h2c_fd, buffer(im, 0))

        print "bytes_written:", bytes_written
        # wait for done
        proc_out = result_queue.get()
        print "got proc out"
        read_process.join()
        print "read process joined"
        result = proc_out


    finally:
        os.close(c2h_fd)
        os.close(h2c_fd)

    if RUN_FPGA:
        if FIXEDP:
            new_result = [k for k in [result[i : i + 8] for i in range(0, len(result), 8)]]
            result = [utils.from_bytestr(k) for k in new_result]
        else:
            result = np.frombuffer(result, dtype=np.float64).reshape(im.shape)

    return np.around(np.array(result),decimals=32) # round for extreme vals


def prepare_flow(reference, flow_tuple, confidence):

    reference_image = np.array(plt.imread(reference, format='png'), dtype=np.float32)*256
    flow_a = np.array(plt.imread(flow_tuple[0], format='png'), dtype=np.float32)*65536

    if TWO_FLOW:
        flow_b = np.array(plt.imread(flow_tuple[1], format='png'), dtype=np.float32)*65536
        flow = np.stack((flow_a,flow_b),axis=-1)
    else:
        flow = flow_a
    flow = np.subtract(flow, 2**15)

    flow = np.divide(flow, 256)

    weight = np.array(plt.imread(confidence, format='png'), dtype=np.float32)*256*256
    weight = np.divide(weight, 65536)

    if VERBOSE:
        print(">>> preparing flow data")
    sz = [reference_image.shape[0], reference_image.shape[1]]

    I_x = np.tile(np.floor(np.divide(np.arange(sz[1]), BILATERAL_SIGMA_SPATIAL)), (sz[0],1))
    I_y = np.tile( np.floor(np.divide(np.arange(sz[0]), BILATERAL_SIGMA_SPATIAL)).reshape(1,-1).T, (1,sz[1]) )
    I_luma = np.floor_divide(utils.rgb2gray(reference_image), float(BILATERAL_SIGMA_LUMA))

    X = np.concatenate((I_x[:,:,None],I_y[:,:,None],I_luma[:,:,None]),axis=2).reshape((-1,3),order='F')
    W0 = np.ravel(weight.T)
    if TWO_FLOW:
        X0 = np.reshape(flow,[-1,2],order='F')
    else:
        X0 = np.reshape(flow,[-1,1],order='F')
    return X, W0, X0, flow_a.shape

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

def dense_solve_jacobi(X, W0, X0):
    if VERBOSE:
        print(">>> starting bilateral grid optimization")

    grid_size = (np.amax(X,axis=0) + np.ones(3)).astype('int64')
    print grid_size

    basis = np.insert(np.cumprod(grid_size)[0:-1],0,1)


    grid_num_vertices = np.prod(grid_size)
    grid_splat_indices = np.dot(X,basis)

    if VERBOSE:
        print(">>> building splat mat")

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

    if TWO_FLOW:
        XW0 = np.column_stack((np.multiply(W0,X0[:, 0]) , np.multiply(W0,X0[:, 1]))).astype('float32')
        b_mat = np.reshape((splat_mat * XW0), np.hstack([grid_size,2]),order='F').astype('float32')
    else:
        XW0 = (W0*X0.T).T.astype('float32')
        b_mat = np.reshape((splat_mat * XW0), grid_size,order='F').astype('float32')

    #% Construct A, b, and c, which define our optimization problem


    c = 0.5 * sum(XW0 * X0,1);

    # initial solution
    inv_a = 1/np.maximum(A_VERT_DIAG_MIN, A_diag)
    #inv_a = np.divide(1,np.maximum(A_VERT_DIAG_MIN, A_diag)).astype('float32')

    final_result_cpu = []
    final_result_fpga = []
    run_fpga_time = 0

    # for each channel in the flow
    if VERBOSE:
        print("starting flow!")

    if TWO_FLOW:
        for channel in range(b_mat.shape[-1]):
            if VERBOSE:
                print("channel", channel)
            b_c = b_mat[:,:,:,channel].astype('float32')
            Y_c = np.multiply(b_c, inv_a)
            Y_c_cpu =  np.multiply(b_c, inv_a)
            Y_c_fpga = np.multiply(b_c, inv_a).astype('float32')
            cpu_time_sequential = 0
            cpu_time_vector = 0
            cpu_time_sequential_compute_only = 0
            fpga_time = 0
            start_vector = time.time()
            if LOOP_OUTSIDE:
                for iter in range(NUM_PCG_ITERS):
                    if VERBOSE:
                        print("iteration", iter)

                    start_vector = time.time()

                    Ysn = np.multiply(Y_c, splat_norm) # flow 1
                    Ysn_reshaped = np.reshape(Ysn,-1,1)
                    Ysn_r2 = np.reshape((diffuse3_mat * Ysn_reshaped), Ysn.shape, order='F')
                    temp_Y = (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 ))

                    #Y_c_cpu = np.multiply((b_c - (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 ) )) , inv_a)
                    Y_c_cpu = np.multiply((b_c - temp_Y) , inv_a)
                    Y_c = Y_c_cpu

                    total_time_vector = time.time() - start_vector #end timing
                    Y_c_cpu = Y_c

                    if VERBOSE:
                        print("\t CPU vector time: \t\t\t", total_time_vector)

                    if RUN_FPGA:
                        np_fpga_input = np.empty(np.prod(grid_size)*5)
                        fpga_elapsed = 0
                        npit = np.nditer(Y_c_fpga, flags=['f_index','multi_index'], order='F')
                        for x in npit:
                            # order: Y, 1/a, bc, splatnorm
                            #print npit.index, "\t", npit.multi_index

                            np_fpga_input[5*npit.index + 0] = Y_c_fpga[npit.multi_index]
                            np_fpga_input[5*npit.index + 1] = splat_norm[npit.multi_index]
                            np_fpga_input[5*npit.index + 2] = b_c[npit.multi_index]
                            np_fpga_input[5*npit.index + 3] = inv_a[npit.multi_index]
                            np_fpga_input[5*npit.index + 4] = Ysn[npit.multi_index]


                        run_fpga_time = time.time()

                        proc_output = invoke_fpga(np_fpga_input)

                        fpga_input = np.array(np_fpga_input).astype('float32')
                        processed_output = invoke_fpga(fpga_input)

                        if NO_GARBAGE_ADDED:
                            Y_c_fpga = processed_output[0::3].reshape(grid_size)
                        else:
                            Y_c_fpga = processed_output[0::10].reshape(grid_size)


            else: # we are looping inside the FPGA, so we send the data once
                start_vector = time.time()
                for iter in range(NUM_PCG_ITERS):
                    Ysn = np.multiply(Y_c_cpu, splat_norm) # flow 1
                    Ysn_reshaped = np.reshape(Ysn,-1,1)
                    Ysn_r2 = np.reshape((diffuse3_mat * Ysn_reshaped), Ysn.shape, order='F')
                    temp_Y = (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 )).astype('float32')

                    Y_c_cpu = np.multiply((b_c - temp_Y) , inv_a).astype('float32')

                total_time_vector = time.time() - start_vector #end timing

                cpu_time_vector += total_time_vector

                if VERBOSE:
                    print("\t CPU vector time: \t\t\t", total_time_vector)

                np_fpga_input = np.empty(np.prod(grid_size)*5+1)
                fpga_elapsed = 0
                npit = np.nditer(Y_c_fpga, flags=['f_index','multi_index'], order='F')
                np_fpga_input[0] = NUM_PCG_ITERS
                for x in npit:
                    # order: Y, 1/a, bc, splatnorm
                    #print npit.index, "\t", npit.multi_index

                    np_fpga_input[5*npit.index + 0+1] = Y_c_fpga[npit.multi_index]
                    np_fpga_input[5*npit.index + 1+1] = splat_norm[npit.multi_index]
                    np_fpga_input[5*npit.index + 2+1] = b_c[npit.multi_index]
                    np_fpga_input[5*npit.index + 3+1] = inv_a[npit.multi_index]
                    np_fpga_input[5*npit.index + 4+1] = Ysn[npit.multi_index]


                if RUN_FPGA:
                    run_fpga_time = time.time()

                    proc_output_2 = invoke_fpga(np_fpga_input)
                    cycles = proc_output_2[0]
                    print "cycles: ", cycles
                    proc_output = proc_output_2[1:]

                    Y_c_fpga = proc_output.reshape(grid_size,order='F')

                fpga_runtime =  time.time() - run_fpga_time
                fpga_time +=fpga_runtime

            final_result_cpu.append(Y_c_cpu)
            final_result_fpga.append(Y_c_fpga)

            if VERBOSE:
                print("Channel",channel,"completed, total time:")
                print("\t CPU vector: \t\t\t", cpu_time_vector)
                print("\t FPGA: \t\t\t", fpga_time)

        final_solution_cpu = np.stack((final_result_cpu[0], final_result_cpu[1]),axis=3)
        final_solution_fpga = np.stack((final_result_fpga[0], final_result_fpga[1]),axis=3)
    else:
        b_c = b_mat[:,:,:].astype('float32')
        Y_c = np.multiply(b_c, inv_a)
        Y_c_cpu =  np.multiply(b_c, inv_a)
        Y_c_fpga = np.multiply(b_c, inv_a).astype('float32')
        Ysn = np.multiply(Y_c, splat_norm)
        cpu_time_sequential = 0
        cpu_time_vector = 0
        cpu_time_sequential_compute_only = 0
        fpga_time = 0
        start_vector = time.time()
        if LOOP_OUTSIDE:
            for iter in range(NUM_PCG_ITERS):
                if VERBOSE:
                    print("iteration", iter)

                start_vector = time.time()

                Ysn = np.multiply(Y_c, splat_norm) # flow 1
                Ysn_reshaped = np.reshape(Ysn,-1,1)
                Ysn_r2 = np.reshape((diffuse3_mat * Ysn_reshaped), Ysn.shape, order='F')
                temp_Y = (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 ))

                #Y_c_cpu = np.multiply((b_c - (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 ) )) , inv_a)
                Y_c_cpu = np.multiply((b_c - temp_Y) , inv_a)
                Y_c = Y_c_cpu

                total_time_vector = time.time() - start_vector #end timing
                Y_c_cpu = Y_c

                if VERBOSE:
                    print("\t CPU vector time: \t\t\t", total_time_vector)

                if RUN_FPGA:
                    np_fpga_input = np.empty(np.prod(grid_size)*5)
                    fpga_elapsed = 0
                    npit = np.nditer(Y_c_fpga, flags=['f_index','multi_index'], order='F')
                    for x in npit:
                        # order: Y, 1/a, bc, splatnorm
                        #print npit.index, "\t", npit.multi_index

                        np_fpga_input[5*npit.index + 0] = Y_c_fpga[npit.multi_index]
                        np_fpga_input[5*npit.index + 1] = splat_norm[npit.multi_index]
                        np_fpga_input[5*npit.index + 2] = b_c[npit.multi_index]
                        np_fpga_input[5*npit.index + 3] = inv_a[npit.multi_index]
                        np_fpga_input[5*npit.index + 4] = Ysn[npit.multi_index]


                    run_fpga_time = time.time()

                    proc_output = invoke_fpga(np_fpga_input)

                    fpga_input = np.array(np_fpga_input).astype('float32')
                    processed_output = invoke_fpga(fpga_input)

                    if NO_GARBAGE_ADDED:
                        Y_c_fpga = processed_output[0::3].reshape(grid_size)
                    else:
                        Y_c_fpga = processed_output[0::10].reshape(grid_size)


        else: # we are looping inside the FPGA, so we send the data once
            start_vector = time.time()
            # for iter in range(NUM_PCG_ITERS):
            #     Ysn = np.multiply(Y_c_cpu, splat_norm) # flow 1
            #     Ysn_reshaped = np.reshape(Ysn,-1,1)
            #     Ysn_r2 = np.reshape((diffuse3_mat * Ysn_reshaped), Ysn.shape, order='F')
            #     temp_Y = (-1 * LOSS_SMOOTH_MULT * np.multiply(splat_norm , Ysn_r2 )).astype('float32')

            #     Y_c_cpu = np.multiply((b_c - temp_Y) , inv_a).astype('float32')

            total_time_vector = time.time() - start_vector #end timing

            cpu_time_vector += total_time_vector

            if VERBOSE:
                print("\t CPU vector time: \t\t\t", total_time_vector)

            np_fpga_input = np.empty(np.prod(grid_size)*5+1)
            fpga_elapsed = 0
            npit = np.nditer(Y_c_fpga, flags=['f_index','multi_index'], order='F')
            np_fpga_input[0] = NUM_PCG_ITERS
            for x in npit:
                # order: Y, 1/a, bc, splatnorm
                #print npit.index, "\t", npit.multi_index

                np_fpga_input[5*npit.index + 0+1] = Y_c_fpga[npit.multi_index]
                np_fpga_input[5*npit.index + 1+1] = splat_norm[npit.multi_index]
                np_fpga_input[5*npit.index + 2+1] = b_c[npit.multi_index]
                np_fpga_input[5*npit.index + 3+1] = inv_a[npit.multi_index]
                np_fpga_input[5*npit.index + 4+1] = Ysn[npit.multi_index]

            print(np_fpga_input.size)

            if RUN_FPGA:
                run_fpga_time = time.time()

                proc_output_2 = invoke_fpga(np_fpga_input)
                cycles = proc_output_2[0]
                print "cycles: ", cycles
                proc_output = proc_output_2[1:]

                else:
                   proc_output = invoke_fpga(np_fpga_input)

                # pdb.set_trace()
                Y_c_fpga = proc_output.reshape(grid_size,order='F')

            fpga_runtime =  time.time() - run_fpga_time
            fpga_time +=fpga_runtime

        final_result_cpu.append(Y_c_cpu)
        final_result_fpga.append(Y_c_fpga)

        if VERBOSE:
            print("completed, total time:")
            print("\t CPU vector: \t\t\t", cpu_time_vector)
            print("\t FPGA: \t\t\t", fpga_time)

        final_solution_cpu = final_result_cpu
        final_solution_fpga = final_result_fpga

    if TWO_FLOW:
        sliced_solution_cpu = (splat_mat.T) * (final_solution_cpu.reshape(grid_num_vertices,-1,order='F'))
        sliced_solution_fpga = (splat_mat.T) * (final_solution_fpga.reshape(grid_num_vertices,-1,order='F'))
    else:
        sliced_solution_cpu = (splat_mat.T) * (final_solution_cpu[0].reshape(grid_num_vertices,-1,order='F'))
        sliced_solution_fpga = (splat_mat.T) * (final_solution_fpga[0].reshape(grid_num_vertices,-1,order='F'))

    return sliced_solution_cpu, sliced_solution_fpga # final result is an array with improved flows in both directions

def make_parser():
    parser = argparse.ArgumentParser(description='Run the bilateral solver on some depth results.')
    parser.add_argument('--fpga', '-f', action='store_true', default=False)
    parser.add_argument('--quiet', '-q', action='store_false', default=False)
    parser.add_argument('--small', '-s', action='store_true')
    parser.add_argument('--fixedpoint', '-fp', action='store_true')
    parser.add_argument('--loopoutside', '-lo', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--nogarb', '-ng', action='store_true', default=False)
    parser.add_argument('--savefig', '-sf', action='store_true')
    parser.add_argument('--display', '-df', action='store_true')

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # loop over test images
    # for imdir in [x[0] for x in os.walk('img/')][1:]:
    #     reference = imdir+'/reference.png'
    #     target = imdir+'/input.png'
    #     confidence = imdir+'/weight.png'

    #imdir = 'img/Adirondack'
    #reference = imdir+'/reference.png'
    #target = imdir+'/input.png'
    #confidence = imdir+'/weight.png'

    # target = 'img/input0.png'
    # reference = 'img/reference.png'
    # confidence = 'img/weight.png'

    input_X, input_W0, input_X0, flow_shape = prepare_flow(reference,(target,), confidence)

    res_cpu, res_fpga = dense_solve_jacobi(input_X, input_W0, input_X0)

    res_cpu = res_cpu.reshape(flow_shape[0], flow_shape[1], -1,order='F')
    res_fpga = res_fpga.reshape(flow_shape[0], flow_shape[1], -1,order='F')


    flow_a = np.array(plt.imread(target, format='png'), dtype=np.float32)
    # flow_b = np.array(plt.imread(flow_b_img, format='png'), dtype=np.float32)

    plt.subplot(3,2,1)
    plt.imshow(flow_a)
    plt.title('flow a input')
    plt.colorbar()

    # plt.subplot(3,2,2)
    # plt.imshow(flow_b)
    # plt.title('flow b input')
    # plt.colorbar()

    plt.subplot(3,2,3)
    plt.imshow(res_cpu[:,:,0])
    plt.title('CPU flow a output')
    plt.colorbar()

    # plt.subplot(3,2,4)
    # plt.imshow(res_cpu[:,:,1])
    # plt.title('CPU flow b output')
    # plt.colorbar()

    plt.subplot(3,2,5)
    plt.imshow(res_fpga[:,:,0])
    plt.title('FPGA flow a output')
    plt.colorbar()

    # plt.subplot(3,2,6)
    # plt.imshow(res_fpga[:,:,1])
    # plt.title('FPGA flow b output')
    # plt.colorbar()
    if SAVE_FIG:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig('myfig'+timestr)
    if DISPLAY_FIG:
        plt.show()


if __name__ == "__main__":
    main()
