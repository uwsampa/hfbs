import struct,pdb
import numpy as np

DATA_SIZE = 64
FRACTIONAL = 48
INTEGRAL = 15

def to_bytestr(num):

    binstr = '0' * 64

    if num >= 0:
        binstr = bin(int(num * 2 ** FRACTIONAL)).replace('0b','').rjust(DATA_SIZE,'0')
    else:
        binstr = bin(int(-2 ** DATA_SIZE + (abs(num) * 2 ** FRACTIONAL))).replace('-0b','')
        if len(binstr) is 65:
            binstr = '0' * 64;

    res = struct.pack('Q', int(binstr,2))
    return res

def from_bytestr(num):

    longnum = struct.unpack('Q',num)[0]
    binstr = bin(longnum).replace('0b','')

    if len(binstr) < DATA_SIZE:
        binstr = '0' * (DATA_SIZE - len(binstr)) + binstr

    result = '0' * DATA_SIZE

    if int(binstr[0],2) is 0:
        fractional_dec = int(binstr[INTEGRAL + 1:],2) / 2.**(len(binstr[INTEGRAL + 1:]))
        result =  (int(binstr[1:INTEGRAL + 1],2) + fractional_dec)
    else:
        result = (int('0b'+binstr,2) - 2 ** DATA_SIZE)/(float(2 ** FRACTIONAL))

    return result

def rgb2gray(rgb):
    x1 = np.rint(0.299 * rgb[...,0]).astype('uint8')
    x2 = np.rint(0.587 * rgb[...,1]).astype('uint8')
    x3 = np.rint(0.114 * rgb[...,2]).astype('uint8')
    return x1+x2+x3
