import numpy as np
from iminuit.util import make_func_code
from scipy import interpolate
from .funcutil import merge_func_code
from ._libstat import integrate1d_with_edges
from .pdf import HistogramPdf


class HistPdf:
    def __init__(self, hy, binedges, xname='x', x_range=()):
        """
        A histogram PDF. User supplies a template histogram with bin contents and bin
        edges. The histogram does not have to be normalized. The resulting PDF is normalized.
        """
        # normalize, so the integral is unity
        yint = hy * (binedges[1:] - binedges[:-1])
        self.hy = hy.astype(float) / float(yint.sum())
        self.binedges = binedges
        if len(binedges) != len(hy) + 1:
            raise ValueError('binedges must be exactly one entry more than hy')
        probs = self.hy * (binedges[1:] - binedges[:-1])
        self._cdfarr = np.concatenate([[0], np.cumsum(probs)])

        # set normalization range
        self.set_range(x_range)

        # Only one variable. The PDF shape is fixed
        varnames = [xname]
        self.func_code = make_func_code(varnames)
        self.func_defaults = None

    def set_range(self, x_range):
        self.x_range = tuple(x_range) if len(x_range) == 2 else (self.binedges[0], self.binedges[-1])
        assert self.x_range[1] > self.x_range[0]
        self.pdf_norm = self._integrate(self.x_range)
        assert self.pdf_norm > 0
        self.cdf_x_min = max(self._cdf(self.x_range[0]), 0.)

    def __call__(self, *arg):
        # normalize pdf over x-range
        return self._eval(*arg) / self.pdf_norm

    def _eval(self, *arg):
        """pdf value"""
        x = arg[0]
        is_array = isinstance(x, (list, tuple, np.ndarray))
        x = np.array(x if is_array else [x])
        idcs = np.digitize(x, self.binedges)
        # fx is zero outside of known bins and outside of specified x range
        fx = np.array([self.hy[i - 1] if i > 0 and i <= len(self.hy) and self.x_range[0] <= xv < self.x_range[1]
                       else 0.0 for i, xv in zip(idcs, x)])
        return fx if is_array else fx[0]

    def integrate(self, bound, nint_subdiv=0, *arg):
        # normalize intergral over x-range
        return self._integrate(bound, nint_subdiv, *arg) / self.pdf_norm

    def _integrate(self, bound, nint_subdiv=0, *arg):
        # nint_subdiv is irrelevant, ignored for analytical integrals.
        # bound usually is smaller than the histogram's bound.
        [ib0, ib1] = np.digitize([bound[0], bound[1]], self.binedges)
        ib0 = min(max(ib0, 0), len(self.binedges) - 1)
        ib1 = min(max(ib1 - 1, 0), len(self.binedges) - 1)
        edge_inside = bound[0] <= self.binedges[ib0] and self.binedges[ib1] <= bound[1]
        middle_edges = self.binedges[ib0:ib1 + 1] if edge_inside else np.array([])
        x = np.concatenate([[bound[0]], middle_edges, [bound[1]]])
        delta_x = np.diff(x)
        fx = self._eval(x)
        return (fx[:-1] * delta_x).sum()

    def cdf(self, *arg):
        # cdf over full bin_edges, then normalized cdf over x-range
        cx = self._cdf(*arg)
        cx = np.maximum(cx - self.cdf_x_min, 0.)
        cx = np.minimum(cx, self.pdf_norm) / self.pdf_norm
        return cx

    def _cdf(self, *arg):
        x = arg[0]
        is_array = isinstance(x, (list, tuple, np.ndarray))
        x = np.array(x if is_array else [x])
        indices = np.digitize(x, self.binedges)
        indices = np.minimum(np.maximum(indices - 1, 0), len(self.binedges) - 1)
        delta_x = x - self.binedges[indices]
        fx = self._eval(x)
        cdf_x = self._cdfarr[indices] + (fx * delta_x)
        return cdf_x if is_array else cdf_x[0]


class MorphHistPdf(object):
    def __init__(self, hist1, hist2, bin_edges1=None, bin_edges2=None, xname='x', x_range=(), nbins=10000):
        # basic checks on inputs
        hist_types = (HistogramPdf, HistPdf, np.ndarray)
        assert isinstance(hist1, hist_types) and isinstance(hist2, hist_types)
        if isinstance(hist1, (HistogramPdf, HistPdf)):
            hist1 = hist1.hy
            bin_edges1 = hist1.binedges
        if isinstance(hist2, (HistogramPdf, HistPdf)):
            hist2 = hist2.hy
            bin_edges2 = hist2.binedges
        assert isinstance(bin_edges1, (np.ndarray, list, tuple)) and isinstance(bin_edges2,
                                                                                (np.ndarray, list, tuple, type(None)))
        bin_edges1 = np.array(bin_edges1)
        if bin_edges2 is None:
            bin_edges2 = bin_edges1
        if len(bin_edges1) != len(hist1) + 1:
            raise ValueError('binedges1 must be exactly one entry more than hist1')
        if len(bin_edges2) != len(hist2) + 1:
            raise ValueError('binedges2 must be exactly one entry more than hist2')

        # normalize entries by binwidth, so the integral is unity
        yint1 = hist1 * (bin_edges1[1:] - bin_edges1[:-1])
        yint2 = hist2 * (bin_edges2[1:] - bin_edges2[:-1])
        pdf1 = hist1.astype(float) / float(yint1.sum())
        pdf2 = hist2.astype(float) / float(yint2.sum())

        # convert shapes into continuous cdfs
        probs1 = pdf1 * (bin_edges1[1:] - bin_edges1[:-1])
        probs2 = pdf2 * (bin_edges2[1:] - bin_edges2[:-1])
        cs1 = np.concatenate([[0], np.cumsum(probs1)])
        cs1 = cs1 / cs1[-1]  # should already be very close to 1, done for np.where.
        start = np.where(cs1 == 0.0)[0][-1]
        end = np.where(cs1 == 1.0)[0][0] + 1
        cs1 = cs1[start:end]
        binning1 = bin_edges1[start:end]
        cs2 = np.concatenate([[0], np.cumsum(probs2)])
        cs2 = cs2 / cs2[-1]  # should already be very close to 1, done for np.where
        start = np.where(cs2 == 0.0)[0][-1]
        end = np.where(cs2 == 1.0)[0][0] + 1
        cs2 = cs2[start:end]
        binning2 = bin_edges2[start:end]
        f_cs1 = interpolate.interp1d(cs1, binning1)
        f_cs2 = interpolate.interp1d(cs2, binning2)

        # scan the two cdfs and store results
        self._y = np.linspace(0, 1, nbins + 1)
        self._x1 = f_cs1(self._y)
        self._x2 = f_cs2(self._y)
        self._prob_unit = 1. / nbins

        # initialize an instance of the morphing function
        self._last_alpha = None
        self.set_alpha(alpha=0.0)

        # set normalization range
        self.set_range(x_range)

        # One observable and one variable
        varnames = [xname, 'alpha']
        self.func_code = make_func_code(varnames)
        self.func_defaults = None

    def set_alpha(self, alpha):
        # initialize an instance of the morphing function
        if alpha != self._last_alpha:
            self._last_alpha = alpha
            self.binedges, self._fxalpha, self._cdfalpha = self._morph_function(self._last_alpha)
            if hasattr(self, 'x_range'):
                self.set_normalization()

    def set_range(self, x_range):
        # default to infinite range
        self.x_range = tuple(x_range) if len(x_range) == 2 else (-1e300, 1e300)
        assert self.x_range[1] > self.x_range[0]
        if hasattr(self, '_last_alpha') and self._last_alpha is not None:
            self.set_normalization()

    def set_normalization(self):
        self.pdf_norm = self._integrate(self.x_range)
        assert self.pdf_norm > 0
        self.cdf_x_min = max(self._cdf(self.x_range[0]), 0.)

    def _morph_function(self, alpha):
        x_alpha = (1. - alpha) * self._x1 + alpha * self._x2
        density = np.concatenate([self._prob_unit / np.diff(x_alpha), [0]])
        fx_alpha = interpolate.interp1d(x_alpha, density, kind='previous', fill_value=(0, 0), bounds_error=False)
        cdf_alpha = interpolate.interp1d(x_alpha, self._y, kind='linear', fill_value=(0, 1), bounds_error=False)
        return x_alpha, fx_alpha, cdf_alpha

    def __call__(self, *arg):
        """pdf value normalized over x-range"""
        if len(arg) > 1:
            self.set_alpha(arg[1])
        # normalize pdf over x-range
        return self._eval(*arg) / self.pdf_norm

    def _eval(self, *arg):
        x = arg[0]
        is_array = isinstance(x, (list, tuple, np.ndarray))
        x = np.array(x if is_array else [x])
        fx = self._fxalpha(x)
        fx = np.array([fv if self.x_range[0] <= xv < self.x_range[1] else 0.0 for xv, fv in zip(x, fx)])
        return fx if is_array else fx[0]

    def integrate(self, bound, nint_subdiv=0, *arg):
        # update alpha?
        if len(arg) > 0:
            self.set_alpha(arg[0])
        # normalize intergral over x-range
        return self._integrate(bound, nint_subdiv, *arg) / self.pdf_norm

    def _integrate(self, bound, nint_subdiv=0, *arg):
        # nint_subdiv is irrelevant, ignored.
        [ib0, ib1] = np.digitize([bound[0], bound[1]], self.binedges)
        ib0 = min(max(ib0, 0), len(self.binedges) - 1)
        ib1 = min(max(ib1 - 1, 0), len(self.binedges) - 1)
        x = [bound[0], self.binedges[ib0], self.binedges[ib1], bound[1]]
        delta_x = np.diff(x)
        fx = self._eval(bound)
        integral = fx[0] * delta_x[0] + fx[-1] * delta_x[-1] + (ib1 - ib0) * self._prob_unit
        return integral

    def cdf(self, *arg):
        # expects x, alpha
        # update alpha?
        if len(arg) > 1:
            self.set_alpha(arg[1])
        # cdf over full bin_edges, then normalized cdf over x-range
        cx = self._cdf(*arg)
        cx = np.maximum(cx - self.cdf_x_min, 0.)
        cx = np.minimum(cx, self.pdf_norm) / self.pdf_norm
        return cx

    def _cdf(self, *arg):
        x = arg[0]
        is_array = isinstance(x, (list, tuple, np.ndarray))
        cx = self._cdfalpha(np.array(x if is_array else [x]))
        return cx if is_array else cx[0]


class MaximumPdf(object):
    def __init__(self, f, g, x_range, nbins=1000):
        self.f, self.g = f, g
        self.func_code, [self.fpos, self.gpos] = merge_func_code(f, g)
        self.arg_length = max(max(self.fpos), max(self.gpos))
        self.set_range(x_range, nbins)
        self.last_arg = None
        self.set_pdf()

    def set_range(self, x_range, nbins):
        assert len(x_range) == 2 and x_range[1] > x_range[0] and nbins > 0
        self.x_space = np.linspace(x_range[0], x_range[1], nbins + 1)
        self.edges = np.array([np.array([low, high]) for low, high in zip(self.x_space[:-1], self.x_space[1:])])
        self.bw = (x_range[1] - x_range[0]) / nbins

    def set_pdf(self, *arg):
        # arguments to pass on to integration.

        # rudimentary caching
        if arg == self.last_arg:
            return

        # leave out x, integrate uses edges.
        farg = tuple([arg[i - 1] for i in self.fpos[1:]]) if len(arg) == self.arg_length and len(arg) > 0 else tuple()
        garg = tuple([arg[i - 1] for i in self.gpos[1:]]) if len(arg) == self.arg_length and len(arg) > 0 else tuple()

        # get cdfs - this can be slow
        f_probs = np.array([integrate1d_with_edges(self.f, edges, self.bw, farg) for edges in self.edges])
        g_probs = np.array([integrate1d_with_edges(self.g, edges, self.bw, garg) for edges in self.edges])
        f_cs = np.cumsum(f_probs)
        assert f_cs[-1] > 0
        f_cs = np.concatenate([[0], f_cs / f_cs[-1]])
        g_cs = np.cumsum(g_probs)
        assert g_cs[-1] > 0
        g_cs = np.concatenate([[0], g_cs / g_cs[-1]])

        # for max pdf multiply the two cdfs
        cumsum = f_cs * g_cs
        probs = np.diff(cumsum)
        density = np.concatenate([probs / self.bw, [0]])
        self._pdf = interpolate.interp1d(self.x_space, density, kind='previous', fill_value=(0, 0), bounds_error=False)
        self._cdf = interpolate.interp1d(self.x_space, cumsum, kind='linear', fill_value=(0, 1), bounds_error=False)

        # done - let's store args for this config
        self.last_arg = arg

    def __call__(self, *arg):
        if len(arg) > 1:
            self.set_pdf(*arg[1:])
        return self._eval(*arg)

    def _eval(self, *arg):
        x = arg[0]
        is_array = isinstance(x, (list, tuple, np.ndarray))
        fx = self._pdf(np.array(x if is_array else [x]))
        return fx if is_array else fx[0]

    def integrate(self, bound, nint_subdiv=0, *arg):
        if len(arg) > 0:
            self.set_pdf(*arg)
        return self._integrate(bound, nint_subdiv, *arg)

    def _integrate(self, bound, nint_subdiv=0, *arg):
        # nint_subdiv is irrelevant, ignored.
        # bound usually is smaller than the histogram's bound.
        # Find where they are:
        [ib0, ib1] = np.digitize([bound[0], bound[1]], self.x_space)
        ib0 = min(max(ib0, 0), len(self.x_space) - 1)
        ib1 = min(max(ib1 - 1, 0), len(self.x_space) - 1)
        edge_inside = bound[0] <= self.x_space[ib0] and self.x_space[ib1] <= bound[1]
        middle_edges = self.x_space[ib0:ib1 + 1] if edge_inside else np.array([])
        x = np.concatenate([[bound[0]], middle_edges, [bound[1]]])
        delta_x = np.diff(x)
        fx = self._pdf(x)
        return (fx[:-1] * delta_x).sum()

    def cdf(self, *args):
        if len(arg) > 1:
            self.set_pdf(*arg[1:])
        return self._cdf(*args)

    def _cdf(self, *arg):
        x = arg[0]
        is_array = isinstance(x, (list, tuple, np.ndarray))
        cx = self._cdf(np.array(x if is_array else [x]))
        return cx if is_array else cx[0]
