import matplotlib.pyplot as plt
from typing import Callable
import numpy as np

class MyArrayMethods:
    """
    MyArrayMethods
    ==============
        This class is a collection of my personal functions that modify given numpy arrays.\n
        I have made this class, because I had hard time keeping track of growing PGM class\n
        since LAB1.

        Owner: Erol Tunahan Gedik
        Version: 0.5.1
        Last Update: 8.12.2022

    """
    @staticmethod
    def map(arr: np.ndarray, func: Callable[[int | float], int]) -> np.ndarray:
        """
        map
        ===
            Applies given function to every element of the array and returns the result array.\n

        Arguments:
        ----------
            arr: np.ndarray
                The array that the given function will be applied.

            func: Callable[[float], float]
                Any function or lambda in the range of the 0..1 that satisfies the following: 
            
                    def func(x: float) -> float
                    ---------------------------

        Returns:
        --------
            np.ndarray
        """
        width, height = arr.shape

        for y in range(1, height-1):
            for x in range(1, width-1):
                arr[x, y] = func(arr[x, y])
        
        return arr

    @staticmethod
    def convol(arr: np.ndarray, window: np.ndarray, point: int = 0) -> np.ndarray:
        """
        convol
        ======
            Calculates the 2-D convolution between given array and window. May be used as convolution filter.

        Arguments:
        ----------
            arr: np.ndarray
                The array that the given window will be applied in convolution
            window: np.ndarray
                The window to apply
        
        Returns:
        --------
            np.ndarray
        """
        result: np.ndarray = np.zeros((arr.shape))

        for y in range(1, arr.shape[0] - 1):
            for x in range(1, arr.shape[1] - 1):                
                temp = 0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        temp += arr[y + j, x + i] * window[j + 1, i + 1]
                result[y, x] = temp + point
        return result

    @staticmethod
    def median(arr: np.ndarray, winsize: tuple = (3, 3)) -> np.ndarray:
        """
        median
        ======
            Replaces the elements with median of their neighbours.

        Arguments:
        ----------
            arr: np.ndarray
                The array that median filter of given window size will be applied.
            winsize: (int, int)
                The window size 
        
        Returns:
        --------
            np.ndarray
        """
        result: np.ndarray = np.zeros((arr.shape), dtype=arr.dtype)
        window: np.ndarray = np.zeros(winsize, dtype=arr.dtype)

        N, M = arr.shape
        n, m = window.shape

        for y in range(n//2, N - n//2):
            for x in range(m//2, M - m//2):                
                for j in range(-n//2, 1 + n//2):
                    for i in range(-m//2, 1 + m//2):
                        window[j + 1, i + 1] = arr[y + j, x + i]
                
                result[y, x] = np.median(window)

        return result

    @staticmethod
    def histogram(arr: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        histogram
        =========
            Makes a histogram with the values of given array.

        Arguments:
        ----------
            arr: np.ndarray
                The given array to find the histogram of
            normalize: bool
                Optional argument to normalize the output, default to False. When normalized, 
                the output may be treated like a probability density function (pdf).

        Returns:
        --------
            np.ndarray
        """
        result: np.ndarray = np.zeros((arr.max()) + 1)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                result[arr[y, x]] += 1
    
        if normalize:
            result /= np.sum(result)
        
        return result

    @staticmethod
    def cdf(arr: np.ndarray) -> np.ndarray:
        """
        cdf
        ===
            Calculates the cumulative distribution function of given pdf (array).

        Arguments:
        ----------
            arr: np.ndarray
                Given array to calculate the cdf.
        
        Returns:
        --------
            np.ndarray
        """
        hist: np.ndarray = MyArrayMethods.histogram(arr, True)
        maxval: int = arr.max()
        result: np.ndarray = np.zeros((maxval + 1), dtype=float)
        
        for i in range(1, maxval + 1):
            result[i] = result[i - 1] + hist[i - 1]
        
        result /= result[-1]
        return result

    @staticmethod
    def hist_eq(arr: np.ndarray) -> np.ndarray:
        """
        hist_eq
        =======
            Calculates histogram equalized image
        
        Arguments:
        ----------
            arr: np.ndarray
                Given array of image

        Returns:
        --------
            np.ndarray
        """
        maxval = arr.max()
        cdf: np.ndarray = MyArrayMethods.cdf(arr)
        result: np.ndarray = np.zeros((arr.shape), dtype=arr.dtype)
        
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                result[y, x] = cdf[arr[y, x]] * maxval

        return result.astype(arr.dtype)

    @staticmethod
    def fft_2d(arr: np.ndarray, shift: bool = True) -> np.ndarray:
        """
        fft_2d
        ======
            Calculates 2D fourier transform of given array using numpy fft

        Arguments:
        ----------
            arr: np.ndarray
                Given 2D array
        
        Returns:
        --------
            np.ndarray
        """
        M, N = arr.shape

        if shift: 
            temp0 = MyArrayMethods.fftshift(arr) # temp0 is shifted copy of input
        else:
            temp0 = np.zeros((M, N)) # temp0 is just zeros

        temp1 = np.zeros((M, N)) # temp1 is for vertical fft of image
        temp2 = np.zeros((M, N)) # temp2 is for horizontal fft of temp1

        for x in range(M):
            temp1[x, :] = np.fft.fft(temp0[x, :], M) / M

        for y in range(N):
            temp2[:, y] = np.fft.fft(temp1[:, y], N) / N

        # return MyArrayMethods.my_dBconverter(temp2)
        return temp2

    @staticmethod
    def fftshift(arr: np.ndarray) -> np.ndarray:
        """
        fftshift
        ========
            Shifts the 2D array center to edges and edges to center, similar to numpy fftshift

        Arguments:
        ----------
            arr: np.ndarray
                Given 2D array
        
        Returns:
        --------
            np.ndarray
        """
        shifted = np.zeros((arr.shape))

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                shifted[x, y] = arr[x, y] * np.power(-1, x + y)
        
        return shifted

    @staticmethod
    def to_decibell(arr: np.ndarray, eps: float = 10e-6) -> np.ndarray:
        """
        to_decibell
        ===========
            Converts given array to dB

        Arguments:
        ----------
            arr: np.ndarray
                Given array
            eps: float
                A small number to prevent taking log of zero
        
        Returns:
        --------
            np.ndarray
        """
        result: np.ndarray

        result = np.abs(arr)
        result = 20 * np.log2(result + eps)

        return result

    @staticmethod
    def ifft_2d(arr: np.ndarray, **kwargs) -> np.ndarray:
        """
        ifft_2d
        =======
            Takes inverse fft of a 2D array
        
        Arguments:
        ----------
            arr: np.ndarray
                Given array
            kwargs: dict
                dtype=... if result array needs to have a specific type
        
        Returns:
        --------
            np.ndarray 
        """
        M, N = arr.shape

        # Shifting the input array which is 
        # suppose to be in the frequency domain
        # then taking complex conjugate of it
        shifted = MyArrayMethods.fftshift(arr).conj()
        # Taking Fourier Transform of the previous array
        # Taking complex conjugate, then dividing  
        # it to the number of elements it contains
        inverse = M*N * MyArrayMethods.fft_2d(shifted).conj()
        # Taking the abs of the array to get rid of complex parts
        return np.abs(inverse).astype(kwargs.get('dtype', inverse.dtype))

    @staticmethod
    def hough_params(arr: np.ndarray, **kwargs) -> tuple:
        M, N = arr.shape

        # Maximum distance between any two points
        D = int(np.ceil(np.sqrt(M**2 + N**2)))  # ceil for round up

        angles = kwargs.get('angles', 2*D)      # I wanted to have a square image for Hough Transform output
                                                # but it can be overriden by adding angles=... when calling function

        theta = np.linspace(-np.pi/2, np.pi/2, angles)
        rho = np.linspace(-D, D, 2*D)                   

        return (rho, theta)

    @staticmethod
    def hough_space(arr: np.ndarray, **kwargs) -> np.ndarray:
        M, N = arr.shape

        (rho, theta) = MyArrayMethods.hough_params(arr, **kwargs)

        # Hough Accumulator
        acc = np.zeros((len(rho), len(theta)))

        # All non-zero points
        non_zero = []

        for x in range(M):
            for y in range(N):
                if arr[x, y] != 0:
                    non_zero.append((x, y))

        D = int(np.ceil(np.sqrt(M**2 + N**2)))

        for x, y in non_zero:
            for i in range(len(theta)):
                # p = x*cos(theta) + y*sin(theta) -> rho is in range of [-D, D]
                # since we don't want negative indexes, we must shift it by D
                p = np.round(x*np.cos(theta[i]) + y*np.sin(theta[i])) + D
                acc[int(p), i] += 1
        
        return acc
