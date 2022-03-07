# Scripts for creating continuous scanning probe patterns
# Colin Ophus - 2022-03-06

import numpy as np
import matplotlib.pyplot as plt
from typing import Union

class ScanPattern:
    '''
    Class for holding scan pattern data.
    '''

    def __init__(
        self,
        scan_size = (1024, 1024),
        step_ratio = np.sqrt(2), #(np.sqrt(5)+1)/2.0,
        step_size = 1.0,
        num_terms = 1,
        ):
        '''
        Args:
            scan_size (array):      Field of view size in pixels.
            step_ratio (float):     Ratio of x to y step size.
            step_size (float):      Step size in pixels.
            num_terms (int):        Number of terms in polynomial series.  Setting to 1 returns
                                    a typical Lissajous curve, while a higher value approximates
                                    a Billiard ball scan (usually 16 or 32 is sufficient).

        '''

        # initialize all terms
        self.scan_size = np.array(scan_size)
        self.step_ratio = float(step_ratio)
        self.step_size = float(step_size)
        self.num_terms = int(num_terms)

        # scan coefficients
        self.scan_coefs = np.array([1.0,step_ratio])
        self.scan_coefs /= np.linalg.norm(self.scan_coefs)
        self.scan_coefs *= step_size * np.pi / self.scan_size


    def generate_positions(
        self,
        start=0.0,
        stop=None,
        step=1.0,
        ):
        '''
        Calculate all scan positions for a given time range.

        If stop is set to None, return 
        '''

        # Scan size if user doesn't provide it
        if stop is None:
            stop = np.prod(self.scan_size)

        # time steps
        self.t = np.arange(start,stop,step)

        # Lissajou pattern
        self.x = np.sin(self.scan_coefs[0]*self.t)
        self.y = np.sin(self.scan_coefs[1]*self.t)
        scale = 1.0

        # Loop for additional terms
        for a0 in range(1,self.num_terms):
            self.x += (((-1)**a0)/(2*a0+1)**2)*np.sin((self.scan_coefs[0]*(2*a0+1))*self.t)
            self.y += (((-1)**a0)/(2*a0+1)**2)*np.sin((self.scan_coefs[1]*(2*a0+1))*self.t)
            scale += 1.0 / (2*a0+1)**2

        # Scaling
        self.x *= ((self.scan_size[0]-2) / 2 / scale)
        self.y *= ((self.scan_size[1]-2) / 2 / scale)

        # Centering
        self.x += self.scan_size[0]/2 - 1/2
        self.y += self.scan_size[1]/2 - 1/2



    def plot_scan(
                self,
            figsize: Union[list, tuple, np.ndarray] = (8, 8),
        ):
        '''
        Plot the scanlines as a vector plot
        '''

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.plot(
            self.y,
            self.x,
            )
        ax.set_xlim([-1,self.scan_size[1]])
        ax.set_ylim([-1,self.scan_size[0]])
        ax.invert_yaxis()

        plt.show()


    def density_image(
            self,
            plotfig=True,
            cmap='viridis',
            figsize: Union[list, tuple, np.ndarray] = (10,10),
        ):
        '''
        Generate an image with the scan line density. Optionally plot the result. 
        Returns the reuslting image.
        '''

        xF = np.floor(self.x).astype('int')
        yF = np.floor(self.y).astype('int')
        dx = self.x - xF
        dy = self.y - yF

        image_density = \
            np.bincount(
            np.ravel_multi_index((xF  ,yF  ), self.scan_size),
            weights=(1-dx)*(1-dy),
            minlength=np.prod(self.scan_size),
            ) + \
            np.bincount(
            np.ravel_multi_index((xF+1,yF  ), self.scan_size),
            weights=(  dx)*(1-dy),
            minlength=np.prod(self.scan_size),
            ) + \
            np.bincount(
            np.ravel_multi_index((xF  ,yF+1), self.scan_size),
            weights=(1-dx)*(  dy),
            minlength=np.prod(self.scan_size),
            ) + \
            np.bincount(
            np.ravel_multi_index((xF+1,yF+1), self.scan_size),
            weights=(  dx)*(  dy),
            minlength=np.prod(self.scan_size),
            )

        image_density = np.reshape(image_density, self.scan_size)

        if plotfig is True:
            # image_plot = image_density - np.mean(image_density)
            # image_plot = image_plot / np.std(image_plot)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

            ax.imshow(
                image_density,
                cmap=cmap,
                vmin=0,
                vmax=2.0*np.mean(image_density))
            # ax.set_xlim([-1,self.scan_size[1]])
            # ax.set_ylim([-1,self.scan_size[0]])
            # ax.invert_yaxis()

            plt.show()

        return image_density


    def save_points(
        self,
        filename,
        ):
        '''
        Save scan positions in pixel values into a comma separated value ascii text file.
        '''

        np.savetxt(
            filename, 
            np.vstack((self.x,self.y)).T,
            delimiter=',',
            fmt='%1.3f'
            )

            
