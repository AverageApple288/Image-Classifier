import cupy as cp

def max_pool(image, pool_size=2, stride=2):
    img_height, img_width, channels = image.shape
    out_height = (img_height - pool_size) // stride + 1
    out_width = (img_width - pool_size) // stride + 1
    pooled = cp.zeros((out_height, out_width, channels))

    for c in range(channels):
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                w_start = j * stride
                window = image[h_start:h_start+pool_size, w_start:w_start+pool_size, c]
                pooled[i, j, c] = cp.max(window)
    return pooled