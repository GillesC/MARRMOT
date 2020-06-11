import warnings

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)


# Create models from data
def best_fit_distribution(data, bins='auto', ax=None, verbose=False):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    # x is bin_edges so compute center values
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.rayleigh, st.rice, st.norm
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                if verbose:
                    print(f"Fitting {distribution.name}")
                    print(f"\t Loc {loc}")
                    print(f"\t Scale {scale}")
                    print(f"\t SSE {sse}")

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x, name=distribution.name).plot(ax=ax, legend=True)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution, best_params, best_sse, (y, x)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


#
# root_dir = abspath(pjoin(dirname(__file__), ".."))
# env_path = abspath(pjoin(root_dir, "data", "IL", "scanning", "868-ULA"))
# scenario_dir_path = pjoin(env_path, "LL")
# input_channels_path = pjoin(scenario_dir_path, "channels_data_carr_all_positions.ftr")
# df_channels = pd.read_feather(input_channels_path)
#
# # Load data
# # data = df_channels.query("ChannelPower_dB > -13 and ChannelPower_dB < -10").ChannelPower_dB.dropna()
# data = df_channels.ChannelPower_dB.dropna()
#
# # Plot for comparison
# plt.figure(figsize=(12, 8))
# ax = data.plot(kind='hist', density=True, alpha=0.5)
# # Save plot limits
# dataYLim = ax.get_ylim()
#
# # Find best fit distribution
# best_fit, best_fit_params, best_see = best_fit_distribution(data, ax)
# best_fit_name =best_fit.name
#
# best_dist = getattr(st, best_fit_name)
#
# # Update plots
# ax.set_ylim(dataYLim)
# ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
# ax.set_xlabel(u'Temp (°C)')
# ax.set_ylabel('Frequency')
# plt.legend()
# plt.show()
#
#
# # Make PDF with best params
# pdf = make_pdf(best_dist, best_fit_params)
#
# # Display
# plt.figure(figsize=(12, 8))
# ax = pdf.plot(lw=2, label='PDF', legend=True)
# data.plot(kind='hist', density=True, alpha=0.5, label='Data', legend=True, ax=ax)
#
# param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
# param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
# dist_str = '{}({})'.format(best_fit_name, param_str)
#
# ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
# ax.set_xlabel(u'Temp. (°C)')
# ax.set_ylabel('Frequency')
# plt.show()

def get_k_factor(dist_param):
    b = dist_param[0]
    std = dist_param[-1]

    variance = std ** 2

    return b ** 2 / (2 * variance)


def freedman_diaconis(data, return_as="bins"):
    """
        Use Freedman Diaconis rule to compute optimal histogram bin width.
        ``return_as`` can be one of "width" or "bins", indicating whether
        the bin width or number of bins should be returned respectively.


        Parameters
        ----------
        data: np.ndarray
            One-dimensional array.

        return_as: {"width", "bins"}
            If "width", return the estimated width for each histogram bin.
            If "bins", return the number of bins suggested by rule.
        """
    data = np.asarray(data, dtype=np.float_)
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    cube = np.cbrt(data.size)
    bw = (2 * IQR) / cube

    if return_as == "width":
        result = bw
    else:
        rng = data.max() - data.min()
        result = int((rng / bw) + 1)
    return result


def raw_moment(data: np.ndarray, moment=1):
    return np.sum(np.power(data, moment)) / data.size


def fit_moments_rice(data):
    from scipy.optimize import minimize
    from scipy.special import eval_laguerre

    def solve_first_two_moments():
        def min_first_two_moments(param):
            v = param[0]
            sigma = param[1]

            k = v ** 2 / (2 * sigma ** 2)
            a = np.sqrt(np.pi / 2)

            moments = np.zeros(2, dtype=float)
            moments[0] = sigma * a * eval_laguerre(1 / 2, -k)
            moments[1] = 2 * sigma ** 2 + v ** 2

            sample_moments = [raw_moment(data, moment=i + 1) for i in range(2)]
            cost = (np.sum(moments) - np.sum(sample_moments))**2
            return cost

        v0 = np.mean(data)
        s0 = np.std(data)
        res = minimize(min_first_two_moments, np.array([v0, s0]), method="L-BFGS-B", bounds=[(0, None), (0, None)])
        print(res)
        # assert res.success
        return tuple(res.x)

    def min_moments(param):
        v = param[0]
        sigma = param[1]

        k = v ** 2 / (2 * sigma ** 2)
        a = np.sqrt(np.pi / 2)

        moments = np.zeros(6, dtype=float)

        moments[0] = sigma * a * eval_laguerre(1 / 2, -k)
        moments[1] = 2 * sigma ** 2 + v ** 2
        moments[2] = 3 * (sigma ** 3) * a * eval_laguerre(3 / 2, -k)
        moments[3] = 8 * (sigma ** 4) + 8 * (sigma ** 2) * (v ** 2) + (v ** 4)
        moments[4] = 15 * (sigma ** 5) * a * eval_laguerre(5 / 2, -k)
        moments[5] = 48 * (sigma ** 6) + 72 * (sigma ** 4) * (v ** 2) + 18 * (sigma ** 2) * (v ** 4) + (v ** 6)

        sample_moments = np.array([raw_moment(data, moment=i + 1) for i in range(6)])

        weights = [6-i for i in range(6)]
        moments = weights * moments
        sample_moments = weights * sample_moments

        cost = (np.sum(sample_moments) - np.sum(moments))**2
        return cost

    # initial guess based on the first two moments
    v0, s0 = solve_first_two_moments()
    res = minimize(min_moments, np.array([v0, s0]), method="L-BFGS-B", bounds=[(0, None), (0, None)])
    print(res)
    # assert res.success
    return tuple(res.x)


def fit(data, dist, bins=None):
    assert dist is st.rice

    if bins is None:
        bins = freedman_diaconis(data)
    y, x = np.histogram(data, bins=bins, density=True)
    # x is bin_edges so compute center values
    x = (x + np.roll(x, -1))[:-1] / 2.0

    loc = np.min(data)
    data = data - loc
    v, sigma = fit_moments_rice(data)
    scale = sigma
    b = v / scale

    # pdf = dist.pdf(x, loc=loc, scale=scale)

    #
    #
    #
    # # Separate parts of parameters
    # arg = params[:-2]
    # loc = params[-2]
    # scale = params[-1]
    #
    # # Calculate fitted PDF and error with fit in distribution
    # pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
    # sse = np.sum(np.power(y - pdf, 2.0))

    return (scale, b, loc), bins
