import numpy as np
import h5py
import matplotlib.pyplot as plt

def perform_fft_2d(input_array):
    # Perform 2D Fourier transform
    fft_result = np.fft.fft2(input_array)
    
    # Shift the zero frequency component to the center
    fft_result_shifted = np.fft.fftshift(fft_result)

    # Calculate the magnitude spectrum (logarithmic scale for better visualization)
    magnitude_spectrum = np.log(np.abs(fft_result_shifted) + 1)

    return fft_result_shifted, magnitude_spectrum

def plot_fft_result(input_array, fft_result_shifted, magnitude_spectrum):
    # Plot the original image
    plt.subplot(131), plt.imshow(input_array, cmap='RdBu')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # Plot the Fourier Transform (magnitude spectrum)
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='grey')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    # Plot the inverse Fourier Transform (should be close to the original image)
    inverse_fft_result = np.fft.ifft2(np.fft.ifftshift(fft_result_shifted)).real
    plt.subplot(133), plt.imshow(inverse_fft_result, cmap='RdBu')
    plt.title('Inverse Fourier Transform'), plt.xticks([]), plt.yticks([])

    plt.show()

def plot_image(image_data):
    # Convert the 2D list to a NumPy array
    image_array = np.array(image_data)

    # Plot the image using the 'RdBu' colormap
    plt.imshow(image_array, cmap='RdBu', vmin=np.min(image_array), vmax=np.max(image_array))
    plt.colorbar()  # Add a color bar to the right of the plot
    plt.show()


with h5py.File(r"C:\Users\aksha\Downloads\Spinodal-decomposition-main\Spinodal-decomposition-main\fi.h5",'r') as hdf:
    ls=list(hdf.keys())
    #print(ls)
    data=hdf.get('matrix_0')
    #print(type(data))
    matrix1=list(data)
    #print(type(matrix1))
    #print(matrix1)
    
    plot_image(matrix1)
    fft_result_shifted, magnitude_spectrum = perform_fft_2d(np.array(matrix1))
    print(magnitude_spectrum[49])
    plt.plot(magnitude_spectrum[48])
    plt.title('Plot of 100 Random Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    
    
    plot_fft_result(np.array(matrix1), fft_result_shifted, magnitude_spectrum)
   
    