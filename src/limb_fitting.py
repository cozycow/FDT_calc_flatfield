import numpy as np


def find_center(image, repeat=1, **kwargs):
    edges = find_edges(image, **kwargs)
    xe, ye = np.where(edges)
    if len(xe) > 2:
        xc, yc, r = [], [], []
        for _ in range(repeat):
            xf, yf = filter_outliers(xe, ye, **kwargs)
            xc_, yc_, r_ = fitnp(xf, yf)
            xc += [xc_]
            yc += [yc_]
            r += [r_]
        return np.median(xc), np.median(yc), np.median(r)
    else:
        return 0, 0, 0


def find_edges(image, sigma=0, threshold=0.5, low=0.1, high=99.9, **kwargs):
    from skimage.feature import canny
    a = np.percentile(image, low)
    b = np.percentile(image, high)
    threshold_ = (b - a) * threshold

    edges = canny(np.nan_to_num(image), sigma=sigma, low_threshold=threshold_, high_threshold=threshold_)
    return edges


def filter_outliers(x, y, acc=1, max_circles=1000, **kwargs):
    t = np.random.permutation(len(x) // 3 * 3).reshape(3, -1)
    t = t[:,:max_circles]
    xc0, yc0, r0 = fit3p(x[t], y[t])
    inliers = np.abs(np.sqrt((np.expand_dims(x, axis=0) - np.expand_dims(xc0, axis=1)) ** 2 +
                             (np.expand_dims(y, axis=0) - np.expand_dims(yc0, axis=1)) ** 2) -
                     np.expand_dims(r0, axis=1)) < acc

    n_inliers = np.sum(inliers, axis=1)
    best_inliers = np.where(inliers[np.argmax(n_inliers)])
    return x[best_inliers], y[best_inliers]


def fit3p(x, y):
    x1, x2 = x[1] - x[0], x[2] - x[0]
    y1, y2 = y[1] - y[0], y[2] - y[0]

    q = (x1 * y2 - x2 * y1) * 2
    q = q / (q ** 2 + 1e-16)
    a1, a2 = (x1 ** 2 + y1 ** 2) * q, (x2 ** 2 + y2 ** 2) * q

    xc = y2 * a1 - y1 * a2
    yc = x1 * a2 - x2 * a1
    return xc + x[0], yc + y[0], np.sqrt(xc ** 2 + yc ** 2)


def fitnp(x, y):
    mx, my = np.mean(x), np.mean(y)
    s = np.sqrt(np.std(x) ** 2 + np.std(y) ** 2)
    x_, y_ = (x - mx) / s, (y - my) / s

    A = np.array([(x_ ** 2 + y_ ** 2) / 2, x_, y_])
    b = np.sum(A, axis=1)
    q = b @ np.linalg.inv(A @ A.T)

    xc, yc = -q[1] / q[0], -q[2] / q[0]
    r = np.sqrt(xc ** 2 + yc ** 2 + 2 / q[0])
    return xc * s + mx, yc * s + my, r * s
