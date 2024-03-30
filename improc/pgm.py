from .array_methods import MyArrayMethods
from typing import Callable
import numpy as np

class PGM:
    """
    PGM
    ===
        PGM class is my personal .pgm file interface for the ELM463 Image Processing course 

        Owner: Erol Tunahan Gedik
        Version: 0.5.1
        Last Update: 8.12.2022

    Attributes:
    -----------
        data: np.ndarray
            The image itself, stored as a numpy array for extra functionality\n
        maxval: int
            The maximum value a pixel can get\n
        p: str
            The first token of .pgm files. Default to 'P5'.\n
    Methods:
    --------
        PGM(data) -> None
            Default constructor\n
        @staticmethod read(file_path) -> PGM
            .pgm file reader. Also an alternative constructor when reading a file.\n
        write(file_path) -> PGM
            .pgm file writer.\n
        max() -> int
            Returns maximum pixel value since the attribute is private.\n
        map(func) -> PGM
            Maps pixels of the image to a new image using the given function.\n
        median(winsize) -> PGM 
            Applies median filter with given window size.\n
        convol(window) -> PGM
            Calculates the convolution of the image and the given window.\n
        hist_eq() -> PGM
            Changes the dynamic range of pixel values using its histogram.\n
        histogram(normalize) -> ndarray
            Calculates and returns the histogram with the option of normalization.\n
        cdf() -> ndarray
            Calculates and returns the cumulative distribution function.\n
        fft_2d() -> ndarray
            Returns (2D) fourier transformed image using numpy FFT
        to_ndarray() -> ndarray
            Returns the image (data attribute) in the form of numpy array.\n  
    Design Notes:
    -------------
            PGM class may look like one big mess, and it is. It is actually nothing but a wrapper around\n
        a single numpy array. One design choice I want to point out is, most methods return a new instance\n 
        of the PGM class. I chose to do this because this will allow me to chain methods back to back and\n
        write some strong one-liners.\n
    Example:
    --------
    ------------------------------------------------------------------------------------------------------------------
        image = PGM.read('fig-3_20-a.pgm').map(lambda x: 1 - x).hist_eq().to_ndarray()

    ------------------------------------------------------------------------------------------------------------------            
        In this example first a PGM object is created from the file, then negative, then histogram equilization\n
        and finally to numpy array. This way of method chaining was inspired by one of my favorite languages, Rust.\n
        
        Also want to point out that all methods that modify the numpy array in this class can be boiled down\n 
        to single method that takes a function as the parameter. I think that method would be a functor then.\n
    """
    __data: np.ndarray
    __maxval: int
    __p: str

    def __init__(self, data: np.ndarray | list, *args, **kwargs) -> None:
        """
        PGM Constructor
        ===============
            Default constructor of PGM class. 

        Arguments:
        ----------
            data: np.ndarray | list
                Array representation of the .pgm image.

        Returns:
        --------
            None
        """
        if isinstance(data, list):
            self.__data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.__data = data.copy()

        self.__p = kwargs.get('p', 'P5')
        self.__maxval = kwargs.get('maxval', self.__data.max())

    @staticmethod
    def read(file_path: str) -> 'PGM':
        """
        read
        ====
            Static method to read .pgm files. Returns a new PGM instance and 
            can be thought of like an alternative to default constructor.

        Arguments:
        ----------
            file_path: str
                Path to the .pgm file.
            
        Returns:
        --------
            PGM
        """
        with open(file_path, 'rb') as file:
            p = str(file.readline(), 'utf-8')
            
            while True:
                line = file.readline()
                if line[0] != '#':
                    break

            width, height = (int(s) for s in file.readline().split())
            maxval = int(file.readline())
            data = np.empty((height, width), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    data[y, x] = ord(file.read(1))

        return PGM(data, maxval=maxval, p=p)

    def write(self, file_path: str) -> 'PGM':
        """
        write
        =====
            Method to write into .pgm files in correct format.

        Arguments:
        ----------
            file_path: str
                Path to .pgm file.
        
        Return:
        -------
            The same PGM instance
        """
        height, width = self.__data.shape
        with open(file_path, 'wb') as file:
            pgm_head_str = f"{self.__p}\n{height} {width}\n{self.__maxval}\n"
            pgm_head_b = bytearray(pgm_head_str, 'utf-8')
            file.write(pgm_head_b)
            pgm_data = self.to_ndarray().tobytes()
            file.write(pgm_data)

        return self
    
    def max(self) -> int:
        return self.__maxval

    def map(self, func: Callable[[int | float], int]) -> 'PGM':
        return PGM( 
            MyArrayMethods.map(
                self.to_ndarray(), 
                func
        ))

    def median(self, winsize: tuple = (3, 3)) -> 'PGM':
        return PGM(
            MyArrayMethods.median(
                self.to_ndarray(),
                 winsize
        ))

    def convol(self, window: np.ndarray, point: int = 0) -> 'PGM':
        return PGM(
            MyArrayMethods.convol(
                self.to_ndarray(), 
                window,
                point=point
        ))

    def hist_eq(self) -> 'PGM':
        return PGM( MyArrayMethods.hist_eq(self.to_ndarray()) )

    def histogram(self, normalize: bool = False) -> np.ndarray:
        return MyArrayMethods.histogram(self.to_ndarray(), normalize)

    def cdf(self) -> np.ndarray:
        return MyArrayMethods.cdf(self.to_ndarray())

    def fft_2d(self, shift: bool = True) -> np.ndarray:
        return MyArrayMethods.fft_2d(self.to_ndarray(), shift)

    def hough_space(self, **kwargs) -> np.ndarray:
        return MyArrayMethods.hough_space(self.to_ndarray(), **kwargs)

    def to_ndarray(self) -> np.ndarray:
        return self.__data.copy()
