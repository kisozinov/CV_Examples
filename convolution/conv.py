import numpy as np


def convolution(input_tensor, filters, stride=(1, 1)):
    C, H_in, W_in = input_tensor.shape
    _, M, R, S = filters.shape
    H_out = (H_in - R) // stride[0] + 1
    W_out = (W_in - S) // stride[1] + 1
    output_tensor = np.zeros((M, H_out, W_out))
    # By num of filters
    for m in range(M):
        # x
        for x in range(H_out):
            # y
            for y in range(W_out):
                conv_result = 0
                # Channels
                for k in range(C):
                    # Filter size
                    for i in range(R):
                        for j in range(S):
                            conv_result += input_tensor[k, x*stride[0]+i, y*stride[1]+j] * filters[k, m, i, j]
                output_tensor[m, x, y] = conv_result
    return output_tensor


def im2col(input_tensor, filter_height, filter_width, stride=(1, 1)):
    C, H_in, W_in = input_tensor.shape
    H_out = (H_in - filter_height) // stride[0] + 1
    W_out = (W_in - filter_width) // stride[1] + 1
    col_matrix = np.zeros((C * filter_height * filter_width, H_out * W_out))
    col_index = 0
    for i in range(0, H_in - filter_height + 1, stride[0]):
        for j in range(0, W_in - filter_width + 1, stride[1]):
            col_matrix[:, col_index] = input_tensor[:, i:i+filter_height, j:j+filter_width].reshape(-1)
            col_index += 1
    return col_matrix

def conv_layer_im2col(input_tensor, filters, stride=(1, 1)):
    _, H_in, W_in = input_tensor.shape
    C, M, R, S = filters.shape
    H_out = (H_in - R) // stride[0] + 1
    W_out = (W_in - S) // stride[1] + 1
    col_matrix = im2col(input_tensor, R, S, stride)
    output_tensor = np.zeros((M, H_out * W_out))
    for c in range(C):
        for m in range(M):
            filter_matrix = filters[c][m].reshape(1, -1)
            conv_result = np.dot(filter_matrix, col_matrix)
            output_tensor[m] = conv_result
        return output_tensor