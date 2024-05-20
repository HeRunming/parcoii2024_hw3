# Foundations of Parallel Computing II, Spring 2024.
# Instructor: Chao Yang, Xiuhong Li @ Peking University.
import sys
import numpy as np
import cv2

def process_uf_to_color_image(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        first_line = file.readline().strip().split()
        N, M = map(int, first_line)  # N is the number of rows, M is the number of columns

        uf = []
        for i in range(N):
            uf_line = list(map(int, file.readline().strip().split()))
            uf.extend(uf_line)

    uf_array = np.array(uf, dtype=int).reshape(N, M)

    min_indices = {root: N * M for root in set(uf_array.flatten())}

    # Calculate the minimum index for each root node
    for i in range(N):
        for j in range(M):
            root = uf_array[i][j]
            if i * M + j < min_indices[root]:
                min_indices[root] = i * M + j

    # Create a label image based on the minimum indices
    label_image = np.zeros((N, M), dtype=int)
    for i in range(N):
        for j in range(M):
            label_image[i][j] = min_indices[uf_array[i][j]]

    # Create a color image for the connected components
    color_image = np.zeros((N, M, 3), dtype=np.uint8)
    unique_labels = np.unique(label_image)
    for label in unique_labels:
        color = ((label % 256), ((label // 256) % 256), ((label // (256 * 2)) % 256))
        color_image[label_image == label] = color

    cv2.imwrite(output_filename, color_image)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_filename output_filename")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    process_uf_to_color_image(input_filename, output_filename)

# Example usage:
# python visual.py output_1.txt connected_components_image_1.png
