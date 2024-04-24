import numpy as np
from conv import convolution, conv_layer_im2col

def main():
    # Input image
    input_tensor = np.array([[[5, 2, 0, 2, 4],
                            [3, 5, 2, 4, 3],
                            [0, 1, 5, 1, 0],
                            [2, 4, 2, 5, 3],
                            [4, 2, 0, 2, 5]]])

    # Filters
    filters = np.array([
            [[[2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]],
            [[0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]]]
        ])
    print("Filters shape: ", filters.shape)
    output_tensor_conv = convolution(input_tensor, filters)
    print("Результат свертки:")
    print(output_tensor_conv, "\n")
    output_tensor_im2col = conv_layer_im2col(input_tensor, filters)
    print("\nРезультат im2col:")
    print(output_tensor_im2col, "\n")
    print("Проверка:")
    print(np.array_equal(output_tensor_im2col.reshape(filters.shape[1], filters.shape[2], filters.shape[3]), output_tensor_conv))


if __name__ == "__main__":
    main()