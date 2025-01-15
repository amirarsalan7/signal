import matplotlib.pyplot as plt
from image_processing import load_image, compute_fft, compute_dct, compute_dwt

def display_results(filepath):
    img = load_image(filepath)

    # نمایش تصویر اصلی
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # FFT
    fft_spectrum = compute_fft(img)
    plt.subplot(2, 3, 2)
    plt.title("FFT Spectrum")
    plt.imshow(fft_spectrum, cmap='gray')
    plt.axis('off')

    # DCT
    dct_spectrum = compute_dct(img)
    plt.subplot(2, 3, 3)
    plt.title("DCT Spectrum")
    plt.imshow(dct_spectrum, cmap='gray')
    plt.axis('off')

    # DWT
    cA, cH, cV, cD = compute_dwt(img)
    plt.subplot(2, 3, 4)
    plt.title("DWT Approximation (cA)")
    plt.imshow(cA, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("DWT Horizontal (cH)")
    plt.imshow(cH, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("DWT Vertical (cV)")
    plt.imshow(cV, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_results('C:\\Users\\amirh\\OneDrive\\Documents\\signal\\image_process\\Asli\\Ajornama_5.jpg')

