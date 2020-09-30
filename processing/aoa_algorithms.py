import numpy as np
import scipy.linalg as LA
import scipy.signal as ss


# Functions
def array_response_vector(array, theta):
    M = array.shape
    v = np.exp(1j * 2 * np.pi * array * np.sin(theta))
    return v / np.sqrt(M)


def music(cov_mat, K, M, angles):
    array = np.linspace(0, (M - 1) / 2, M)
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain (i.e. points you want to look at)
    _, V = LA.eig(cov_mat)
    Qn = V[:, K:M]
    num_angles = angles.size
    pspectrum = np.zeros(num_angles)
    for i in range(num_angles):
        av = array_response_vector(array, angles[i])
        pspectrum[i] = 1 / LA.norm((Qn.conj().transpose() @ av))
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    peaks, _ = ss.find_peaks(psindB, height=1.35, distance=1.5)
    return pspectrum, psindB, peaks


def esprit(cov_mat, K, M):
    # cov_mat is the signal covariance matrix, K is the number of sources, M is the number of antennas
    _, U = LA.eig(cov_mat)
    S = U[:, 0:K]
    phi = LA.pinv(S[0:M - 1]) @ S[1:M]  # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs, _ = LA.eig(phi)
    return np.arcsin(np.angle(eigs) / np.pi)
