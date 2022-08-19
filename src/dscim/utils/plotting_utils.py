from matplotlib.ticker import SymmetricalLogLocator, FuncFormatter
import numpy as np


class MajorSymLogLocator(SymmetricalLogLocator):
    """
    Function taken from https://github.com/matplotlib/matplotlib/issues/17402
    """

    def __init__(self):
        super().__init__(base=10.0, linthresh=1.0)

    @staticmethod
    def orders_magnitude(vmin, vmax):

        max_size = np.log10(max(abs(vmax), 1))
        min_size = np.log10(max(abs(vmin), 1))

        if vmax > 1 and vmin > 1:
            return max_size - min_size
        elif vmax < -1 and vmin < -1:
            return min_size - max_size
        else:
            return max(min_size, max_size)

    def tick_values(self, vmin, vmax):

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        orders_magnitude = self.orders_magnitude(vmin, vmax)

        if orders_magnitude <= 1:
            spread = vmax - vmin
            exp = np.floor(np.log10(spread))
            rest = spread * 10 ** (-exp)

            stride = 10**exp * (
                0.25 if rest < 2.0 else 0.5 if rest < 4 else 1.0 if rest < 6 else 2.0
            )

            vmin = np.floor(vmin / stride) * stride
            return np.arange(vmin, vmax, stride)

        if orders_magnitude <= 2:
            pos_a, pos_b = np.floor(np.log10(max(vmin, 1))), np.ceil(
                np.log10(max(vmax, 1))
            )
            positive_powers = 10 ** np.linspace(pos_a, pos_b, int(pos_b - pos_a) + 1)
            positive = np.ravel(np.outer(positive_powers, [1.0, 5.0]))

            linear = np.array([0.0]) if vmin < 1 and vmax > -1 else np.array([])

            neg_a, neg_b = np.floor(np.log10(-min(vmin, -1))), np.ceil(
                np.log10(-min(vmax, -1))
            )
            negative_powers = -(
                10 ** np.linspace(neg_b, neg_a, int(neg_a - neg_b) + 1)[::-1]
            )
            negative = np.ravel(np.outer(negative_powers, [1.0, 5.0]))

            return np.concatenate([negative, linear, positive])

        else:

            pos_a, pos_b = np.floor(np.log10(max(vmin, 1))), np.ceil(
                np.log10(max(vmax, 1))
            )
            positive = 10 ** np.linspace(pos_a, pos_b, int(pos_b - pos_a) + 1)

            linear = np.array([0.0]) if vmin < 1 and vmax > -1 else np.array([])

            neg_a, neg_b = np.floor(np.log10(-min(vmin, -1))), np.ceil(
                np.log10(-min(vmax, -1))
            )
            negative = -(10 ** np.linspace(neg_b, neg_a, int(neg_a - neg_b) + 1)[::-1])

            return np.concatenate([negative, linear, positive])


def symlogfmt(x, pos):
    return f"{x:.6f}".rstrip("0")
