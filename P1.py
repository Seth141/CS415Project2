from numpy.polynomial import Polynomial
import numpy as np


def run_ui():
    print("Choose an option:")
    print("1 - Problem 1: Polynomial with given roots")
    print("2 - Problem 2: Polynomial division")
    print("3 - Problem 3: Noise removal from a signal")
    print("4 - Terminate the program")

    choice = input("Enter your choice: ")
    if choice == "1":
        filename = input("Enter the name of the text file containing the vector v: ")
        with open(filename, "r") as file:
            # Read the roots as floating-point numbers.
            roots = [float(num) for num in file.read().split(",")]
            coefficients = Poly(roots)
            print("Coefficients of the polynomial:", coefficients)
    elif choice == "2":
        filename_A = input("Enter the name of the text file containing vector A: ")
        filename_B = input("Enter the name of the text file containing vector B: ")
        with open(filename_A, "r") as file:
            vector_A = np.array([complex(num) for num in file.read().split(",")])
        with open(filename_B, "r") as file:
            vector_B = np.array([complex(num) for num in file.read().split(",")])
            print("Resulting polynomial C(x):", divide(vector_A, vector_B))
    elif choice == "3":
        filename = input("Enter the name of the file containing the signal: ")
        threshold = float(input("Enter the threshold value: "))
        with open(filename, "r") as file:
            signal = np.array([complex(num) for num in file.read().split(",")])
            fft_signal = FFT(signal)
            fft_signal[np.abs(fft_signal) < threshold] = 0
            cleaned_signal = IFFT(fft_signal)
            print("Cleaned signal:", cleaned_signal)
    elif choice == "4":
        print("Program terminated.")
        return
    else:
        print("Invalid option selected.")


def Poly(roots):
    # Start with the polynomial "x - roots[0]"
    p = Polynomial([-roots[0], 1])
    # Multiply by "x - root" for each remaining root
    for root in roots[1:]:
        p *= Polynomial([-root, 1])
    # The coefficients are in ascending order, so reverse them
    coeffs = p.coef[::-1]
    # Round and format the coefficients
    formatted_coeffs = []
    for coeff in coeffs:
        # If the coefficient is close to an integer, round it
        if abs(coeff - round(coeff)) < 0.0001:
            coeff = round(coeff)
        formatted_coeffs.append(coeff)
    return formatted_coeffs


# The divide and FFT functions would remain the same

if __name__ == "__main__":
    run_ui()
