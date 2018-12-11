# coding=utf-8

import os
import functools
import operator
import gzip
import struct
import array
import numpy


def parse_data(fd):
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise Exception('Invalid IDX file, '
                        'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise Exception('Invalid IDX file, '
                        'file must start with two zero bytes. '
                        'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise Exception('Unknown data type '
                        '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise Exception('IDX file has wrong number of items. '
                        'Expected: %d. Found: %d' % (expected_items,
                                                     len(data)))

    return numpy.array(data).reshape(dimension_sizes)


def parse_mnist_file(file_path):
    fopen = gzip.open if os.path.splitext(file_path)[1] == '.gz' else open
    with fopen(file_path, 'rb') as fd:
        return parse_data(fd)
