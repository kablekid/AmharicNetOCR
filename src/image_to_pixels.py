import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = 'data\ሀ.png'  
image = Image.open(image_path)




gray_image = image.convert('L')  # 'L' mode is for grayscale
gray_array = np.array(gray_image)

# Step 3: Extract the 29x29 region starting from (1, 1)inclusive
a,b,c,d = 1,28,1,28
for row in range(30):
    for column in range(7):
        square_region = gray_array[a:b, c:d]
        print(f"a: {a} b: {b} c: {c} d: {d}")
        print(square_region)
        # Step 4: Plot the grayscale image
        plt.imshow(square_region, cmap='gray')
        plt.axis('off')  # Hide the axis
        plt.title('29x29 Square Region from ')
        plt.show()
        c+=29
        d+=29
    a+=29
    b+=29
    c,d = 1,28
