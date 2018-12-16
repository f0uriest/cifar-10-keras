def im2col(x, filter_height, filter_width):
    # adapted from Stanford CS231n
    C, H, W = x.shape
    out_height = H-filter_height + 1
    out_width = W - filter_width + 1

    i0 = np.tile(np.repeat(np.arange(filter_height), filter_width), C)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols
