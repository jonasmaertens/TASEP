# https://github.com/sumartoyo/hasel
import numpy as np
import threading
import multiprocessing
import math


def hsl2rgb(hsl: np.ndarray, astype: type = np.float32):
    """
    Convert HSL to RGB color space
    Args:
        hsl (numpy.ndarray): HSL color space image array with shape (height, width, 3)
        astype: data type of the output image array

    Returns:
        numpy.ndarray: RGB color space image array with shape (height, width, 3)
    """

    def core(_hsl, _frgb):

        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]
        fr, fg, fb = _frgb[:, :, 0], _frgb[:, :, 1], _frgb[:, :, 2]

        q = np.zeros(l.shape, dtype=float)

        lbot = l < 0.5
        q[lbot] = l[lbot] * (1 + s[lbot])

        ltop = lbot == False
        l_ltop, s_ltop = l[ltop], s[ltop]
        q[ltop] = (l_ltop + s_ltop) - (l_ltop * s_ltop)

        p = 2 * l - q
        q_sub_p = q - p

        is_s_zero = s == 0
        l_is_s_zero = l[is_s_zero]
        per_3 = 1. / 3
        per_6 = 1. / 6
        two_per_3 = 2. / 3

        def calc_channel(channel, t):
            t[t < 0] += 1
            t[t > 1] -= 1
            t_lt_per_6 = t < per_6
            # don't change to is False or not because it will be a numpy array
            t_lt_half = (t_lt_per_6 == False) * (t < 0.5)
            t_lt_two_per_3 = (t_lt_half == False) * (t < two_per_3)
            t_mul_6 = t * 6

            channel[:] = p.copy()
            channel[t_lt_two_per_3] = p[t_lt_two_per_3] + q_sub_p[t_lt_two_per_3] * (4 - t_mul_6[t_lt_two_per_3])
            channel[t_lt_half] = q[t_lt_half].copy()
            channel[t_lt_per_6] = p[t_lt_per_6] + q_sub_p[t_lt_per_6] * t_mul_6[t_lt_per_6]
            channel[is_s_zero] = l_is_s_zero.copy()

        calc_channel(fr, h + per_3)
        calc_channel(fg, h.copy())
        calc_channel(fb, h - per_3)

    frgb = np.zeros(hsl.shape, dtype=float)
    cpus = multiprocessing.cpu_count()
    length = int(math.ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(hsl[line:line_next], frgb[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return (frgb * 255).round().astype(astype)
