# Qwirkle Game Monitor

**Author:** Gheorghe Bogdan-Alexandru (Group 343)

## Project Overview
The goal of this project is to build a software system capable of automatically calculating the score for a Qwirkle game. The main challenge addressed is enabling the computer to recognize game pieces (shapes and colors) even if the camera angle is imperfect or the pieces are slightly misplaced.

## Methodology

### 1. Image Preparation & Board Isolation
To detect pieces accurately, the system first isolates the game board from the background noise found in phone photos:
* **Preprocessing:** The image is converted to Grayscale and Gaussian blur is applied to remove small details like dust.
* **Grid Detection:** Adaptive Thresholding highlights the grid lines. The program then detects edges and looks for the largest square shape to isolate the board.
* **Perspective Correction:** A `warpPerspective` function stretches the image to create a perfect top-down view of the 16x16 grid.

**The Padding Solution**
To solve the issue where pieces placed on the edge of the board were cut off, the system generates two versions of the board image:
1.  **Standard View:** A 1600x1600 pixel image used to check grid alignment.
2.  **Padded View:** A larger image with an extra **25-pixel border**, allowing the computer to see pieces sticking out of the grid boundaries.

### 2. Detecting Moves
The system tracks game progress by comparing the current photo with the photo from the previous turn:
* The two images are subtracted.
* Pixels changing from black to colorful create white spots in the difference image.
* If a grid square contains enough white spots, the system registers that a piece was placed there.

### 3. Piece Recognition
Once a new piece is found, the system identifies its Shape and Color.

**Shape Recognition (Template Matching)**
* The system uses 6 manually created shapes: **Circle, Square, Diamond, Clover, 4-Point Star, and 8-Point Star**.
* To handle centering issues, squares are analyzed with padding to find the highest match value where the piece is centered. The best match score is kept.

**Color Recognition**
* The system uses the **HSV color mode** to separate Color (Hue) from Brightness (Value), which helps handle shadows.
* Ranges are defined for **Red (both high and low values), Orange, Yellow, Green, Blue, and White**.

### 4. Score Calculation
The program calculates points based on the official Qwirkle rules:
* It scans the board horizontally and vertically starting from the new pieces.
* **Scoring Rules:**
    * Points are added if a line is longer than 1 piece.
    * A **"Qwirkle Bonus" (x2)** is applied if a line contains 6 pieces.
    * It checks for specific board bonuses (like double points) if the configuration includes them.
* **Implementation Detail:** For vertical scanning, the matrix is transposed (rows swapped with columns) to reuse the horizontal scanning function, simplifying the code.
