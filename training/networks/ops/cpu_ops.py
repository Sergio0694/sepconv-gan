import numpy as np

def sepconv(image, kv, kh):
    '''Executes the forward pass of the sepconv operation.

    image(np.array) -- the [n, y, x, 3] input image
    kv(np.array) -- the [n, y, x, kc] vertical kernel
    kh(np.array) -- the [n, y, x, kc] horizontal kernel
    '''

    assert image.ndim == 4
    assert image.shape[0] == kv.shape[0] == kh.shape[0]
    assert image.shape[1] == kv.shape[1] == kh.shape[1]
    assert image.shape[2] == kv.shape[2] == kh.shape[2]
    assert kv.shape[3] == kh.shape[3]
    assert kv.shape[3] % 2 == 1

    output = np.zeros_like(image)
    kc, kc_2 = kv.shape[3], kv.shape[3] // 2
    for n in range(image.shape[0]):
        for c in range(image.shape[3]):
            for y in range(image.shape[1]):
                for x in range(image.shape[2]):
                    pixel = 0.0
                    for i in range(kc):
                        for j in range(kc):
                            y_t, x_t = y - kc_2 + i, x - kc_2 + j
                            if y_t < 0 or y_t >= image.shape[1] or \
                                x_t < 0 or x_t >= image.shape[2]:
                                continue
                            pixel += image[n, y_t, x_t, c] * kv[n, y, x, i] * kh[n, y, x, j]
                    output[n, y, x, c] = pixel
    return output
