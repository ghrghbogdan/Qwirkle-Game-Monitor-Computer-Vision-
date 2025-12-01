# Qwirkle Game Monitor

[cite_start]**Author:** Gheorghe Bogdan-Alexandru (Group 343) [cite: 2]

## Project Overview
[cite_start]The goal of this project is to build a software system capable of automatically calculating the score for a Qwirkle game[cite: 4]. [cite_start]The main challenge addressed by this application is making the computer recognize game pieces (shapes and colors) accurately, even under imperfect camera angles or when pieces are slightly misaligned[cite: 5].

## How It Works

### 1. Image Preparation & Board Isolation
Before detecting pieces, the system isolates the game board from background noise:
* [cite_start]**Preprocessing:** The image is converted to Grayscale and a Gaussian blur is applied to remove small details like dust[cite: 10].
* [cite_start]**Grid Detection:** Adaptive Thresholding is used to highlight grid lines, followed by finding the largest square shape to isolate the board[cite: 11, 12].
* [cite_start]**Perspective Correction:** Since photos are taken from an angle, a `warpPerspective` function stretches the image into a perfect top-down 16x16 grid[cite: 14, 15].

**The Padding Solution**
[cite_start]To solve the issue of pieces on the edge being cut off, two versions of the board image are generated[cite: 16, 17]:
1.  [cite_start]**Standard View:** A 1600x1600 pixel image for checking grid alignment[cite: 18].
2.  [cite_start]**Padded View:** A larger image with an extra 25-pixel border, allowing the system to see pieces extending beyond the grid boundaries[cite: 19, 20].

### 2. Detecting Moves
[cite_start]The system identifies player moves by comparing the current photo with the photo from the previous turn[cite: 22, 24].
* The images are subtracted. [cite_start]If a pixel changes from black to colorful, it results in a white spot[cite: 24, 25].
* [cite_start]If a grid square contains enough white spots, it is registered as a placed piece[cite: 26].

### 3. Piece Recognition
[cite_start]Once a new piece is detected, the system determines its Shape and Color[cite: 28].

**Shape Recognition (Template Matching)**
* [cite_start]The system recognizes 6 manual shapes: Circle, Square, Diamond, Clover, 4-Point Star, and 8-Point Star[cite: 30].
* [cite_start]To handle centering issues, squares are taken with padding, ensuring template matching finds the highest value where the piece is centered[cite: 31, 32]. [cite_start]The best match score is kept[cite: 33].

**Color Recognition**
* [cite_start]**HSV Color Mode:** Used to separate "Color" (Hue) from "Brightness" (Value), helping the system handle shadows on the board[cite: 35, 36].
* [cite_start]**Ranges:** Defined for Red (high and low values), Orange, Yellow, Green, Blue, and White[cite: 36].

### 4. Score Calculation
[cite_start]The algorithm calculates points based on official Qwirkle rules[cite: 38]:
* [cite_start]Scans horizontally and vertically from the new pieces to count connected lines[cite: 39, 40].
* **Scoring Rules:**
    * [cite_start]Points are added for lines longer than 1 piece[cite: 41].
    * [cite_start]A "Qwirkle Bonus" (x2) is applied for lines of 6 pieces[cite: 42].
    * [cite_start]Board bonuses (e.g., double points) are checked based on game configuration[cite: 43].
* [cite_start]**Optimization:** Vertical scanning is performed by transposing the matrix (swapping rows with columns) and reusing the horizontal scanning logic[cite: 44].
