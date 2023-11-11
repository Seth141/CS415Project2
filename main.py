##############################################
# By Seth N. and Adam E.
#
# All problems are done and fully functional.
# Please make sure the input vectors are in
# the format: 1.4, 5.7, 8.0 with the square
# brackets removed so the UI function can read
# them in properly
##############################################

import time
import numpy as np
from numpy.polynomial import Polynomial


def FFT(a):
    n = len(a)
    if n == 1:
        return a
    else:
        a_even = FFT(a[::2])
        a_odd = FFT(a[1::2])
        factor = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate(
            [a_even + factor[: n // 2] * a_odd, a_even + factor[n // 2 :] * a_odd]
        )


def IFFT(a):
    n = len(a)
    a_conjugate = np.conjugate(a)
    fft_conjugate = FFT(a_conjugate)
    return np.conjugate(fft_conjugate) / n


def pad_to_length(a, length):
    return np.pad(a, (0, length - len(a)), "constant")


def divide(A, B):
    # Find the next power of 2 that is greater than or equal to the length of A + length of B - 1
    length = 2 ** np.ceil(np.log2(len(A) + len(B) - 1)).astype(int)
    A_padded = pad_to_length(A, length)
    B_padded = pad_to_length(B, length)

    fft_A = FFT(A_padded)
    fft_B = FFT(B_padded)

    # To avoid division by zero, add a small epsilon where B is zero
    fft_B += np.where(fft_B == 0, 1e-10, 0)

    fft_C = fft_A / fft_B  # Element-wise division

    C = IFFT(fft_C)
    C = np.round(C, decimals=5).real  # Round to 5 decimal places

    # Trim both leading and trailing zeros since we only want the significant part of C
    C = np.trim_zeros(C, "f")
    C = np.trim_zeros(C, "b")

    return C.tolist()


def Poly(roots):
    p = Polynomial([-roots[0], 1])
    for root in roots[1:]:
        p *= Polynomial([-root, 1])
    coeffs = p.coef[::-1]
    formatted_coeffs = [
        round(coeff, 5) if abs(coeff - round(coeff)) < 0.0001 else coeff
        for coeff in coeffs
    ]
    return formatted_coeffs


def remove_noise(signal, threshold):
    fft_signal = FFT(signal)
    fft_signal[np.abs(fft_signal) < threshold] = 0

    # Getting the cleaned signal here:
    cleaned_signal = IFFT(fft_signal)
    cleaned_signal = np.clip(cleaned_signal.real, 0, 1)

    return cleaned_signal


def run_ui():
    print("Choose an option:")
    print("1 - Problem 1: Polynomial with given roots")
    print("2 - Problem 2: Polynomial division")
    print("3 - Problem 3: Noise removal from a signal")
    print("4 - Terminate the program")

    choice = input("Enter your choice: ")
    start_time = time.process_time()  # Start the timer

    if choice == "1":
        filename = input("Enter the name of the text file containing the vector v: ")
        with open(filename, "r") as file:
            roots = [float(num) for num in file.read().split(",")]
            coefficients = Poly(roots)
            print("Coefficients of the polynomial:", coefficients)
    elif choice == "2":
        filename_A = input("Enter the name of the text file containing vector A: ")
        filename_B = input("Enter the name of the text file containing vector B: ")
        with open(filename_A, "r") as file:
            vector_A = [float(num) for num in file.read().split(",")]
        with open(filename_B, "r") as file:
            vector_B = [float(num) for num in file.read().split(",")]
            vector_C = divide(vector_A, vector_B)
            print("Resulting polynomial C(x):", vector_C)
    elif choice == "3":
        filename = input("Enter the name of the file containing the signal: ")
        threshold = float(input("Enter the threshold value: "))
        with open(filename, "r") as file:
            signal = [float(num) for num in file.read().split(",")]
            cleaned_signal = remove_noise(signal, threshold)
            print("Cleaned signal:", cleaned_signal)
    elif choice == "4":
        print("Goodbye!")
        return
    else:
        print("Invalid option selected.")

    end_time = time.process_time()  # This stops the timer
    cpu_time = end_time - start_time
    print(f" \n CPU Time for problem {choice}: {cpu_time} seconds \n")


if __name__ == "__main__":
    run_ui()
