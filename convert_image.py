# Foundations of Parallel Computing II, Spring 2024.
# Instructor: Chao Yang, Xiuhong Li @ Peking University.
import sys
import cv2

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py input_image_path output_binary_image_path output_pixels_file_path")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_binary_path = sys.argv[2]
    output_pixels_path = sys.argv[3]

    # Read the color image
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"Error: The image at {input_image_path} was not loaded correctly.")
        return

    # Convert the image to a grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set a threshold and convert the grayscale image to a binary image
    # Use Otsu's method to automatically determine the optimal threshold value
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(output_binary_path, binary_image)

    with open(output_pixels_path, 'w') as file:
        rows, cols = binary_image.shape

        # First, write the number of rows and columns
        file.write(f"{rows} {cols}\n")

        # Then iterate over each pixel
        for y in range(rows):
            for x in range(cols):
                # Write the pixel value to the file
                file.write(str(binary_image[y, x]))
                if x < cols - 1:
                    file.write(' ')
            file.write('\n')

    print(f"Binary pixel values have been written to {output_pixels_path}.")

if __name__ == "__main__":
    main()

# Example usage:
# python convert_image.py your_file.png binary_image_1.jpg pixels_1.txt
