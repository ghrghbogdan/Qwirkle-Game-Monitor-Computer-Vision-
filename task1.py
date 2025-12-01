import cv2 as cv
import numpy as np
import os

data_file_path = '../evaluare/fake_test/' # calea catre folderul cu imagini de test
output_dir = '../evaluare/fake_test/fisiere_solutie/343_Gheorghe_Bogdan' # calea catre folderul unde se vor salva fisierele text cu rezultatele

config_piece=cv.imread('table_quarter.jpg')
templates_folder='templates'
files=os.listdir(templates_folder)
templates=[]
files.sort()
for file in files:
    if file[-3:]=='png':
        temp=cv.imread(os.path.join(templates_folder,file))
        templates.append(temp)

custom_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

custom_matrix_rot = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1 ,0 ,0 ,0 ,2 ,0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def extrage_careu(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (25, 25), 0)
    # show_image('Blurred', blurred)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # show_image('Adaptive Thresh', thresh)

    kernel = np.ones((20, 20), np.uint8)
    edges_dilated = cv.dilate(thresh, kernel, iterations=1)
    # show_image('Dilated Edges', edges_dilated)
    contours, _ = cv.findContours(edges_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 1600
    height = 1600
    padding = 25
    
    image_copy = image.copy()
    cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    # show_image("detected corners",image_copy)

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    

    dst_padded = np.array([
        [padding, padding],                     # stanga sus
        [width + padding, padding],             # dreapta sus
        [width + padding, height + padding],    # dreapta jos
        [padding, height + padding]             # stanga jos
    ], dtype="float32")

    M_padded = cv.getPerspectiveTransform(puzzle, dst_padded)
    result_padded = cv.warpPerspective(image, M_padded, (width + padding * 2, height + padding * 2))
    
    return result, result_padded

def extract_initial_config(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    graypiece = cv.cvtColor(config_piece, cv.COLOR_BGR2GRAY)
    piece_rot = cv.rotate(graypiece, cv.ROTATE_90_CLOCKWISE)
    left_oriented = cv.absdiff(gray, graypiece)
    right_oriented = cv.absdiff(gray, piece_rot)
    if np.sum(left_oriented) < np.sum(right_oriented):
        return np.array([[0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0]]), custom_matrix_rot
    else:
        return np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]), custom_matrix

def new_config_matrix(prev,current):

    cell_size=100
    diff_threshold=50

    gray_prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    gray_current = cv.cvtColor(current, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(gray_prev, gray_current)

    _, bin_diff = cv.threshold(diff, diff_threshold, 255, cv.THRESH_BINARY)
    kernel = np.ones((9,9), np.uint8)
    
    bin_diff = cv.morphologyEx(bin_diff, cv.MORPH_OPEN, kernel)
    bin_diff = cv.morphologyEx(bin_diff, cv.MORPH_DILATE, kernel)
    # show_image('Diff Thresh', bin_diff)

    for i in range(16):
        for j in range(16):
            cell_diff = bin_diff[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            non_zero_count = cv.countNonZero(cell_diff)
            if non_zero_count > (cell_size * cell_size) // 2.2:
                full_config_matrix[i][j] = 1
    return full_config_matrix
            
def extract_template(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # _, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, grey = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    max_score = -1
    best=-1

    for i, temp in enumerate(templates):
        score= cv.matchTemplate(gray, cv.cvtColor(temp, cv.COLOR_BGR2GRAY), cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(score)
        if max_val > max_score:
            max_score = max_val
            best=i

    return best
    
def extract_color(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    color_ranges = {
        'red':      ((0, 60, 50), (6, 255, 255)),
        'red_high': ((174, 60, 50), (180, 255, 255)),
        'orange':   ((7, 60, 50), (25, 255, 255)),
        'yellow':   ((26, 60, 50), (35, 255, 255)),
        'green':    ((36, 60, 50), (85, 255, 255)),
        'blue':     ((86, 60, 50), (130, 255, 255)),
        'white':    ((0, 0, 180), (180, 50, 255))
    }

    color_count = {color: 0 for color in color_ranges.keys()}

    for color, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv.inRange(hsv, lower_np, upper_np)
        color_count[color] = cv.countNonZero(mask)

    detected_color = max(color_count, key=color_count.get)
    return detected_color


def get_line_score(matrix, i, c):
    length = 1
    
    j = c - 1
    while j >= 0 and matrix[i][j] == 1:
        length += 1
        j -= 1
        
    j = c + 1
    while j < 16 and matrix[i][j] == 1:
        length += 1
        j += 1
        
    score = length
    if length == 6:
        score *= 2
        
    return score if length > 1 else 0

def calculate_score(matrix, prev, add_points):
    new_pieces = matrix - prev
    score = 0
    
    for i in range(16):
        for j in range(16):
            if new_pieces[i][j] == 1 and add_points[i][j] != 0:
                score += add_points[i][j]
    # print(score, end=" ")

    processed_rows = set()
    for i in range(16):
        for j in range(16):
            if new_pieces[i][j] == 1:
                if i not in processed_rows:
                    pts = get_line_score(matrix, i, j)
                    if pts > 0:
                        score += pts
                        processed_rows.add(i)
    # print(score, end=" ")

    matrix_T = matrix.T
    new_pieces_T = new_pieces.T
    processed_cols = set()
    
    for i in range(16):
        for j in range(16):
            if new_pieces_T[i][j] == 1:
                if i not in processed_cols:
                    pts = get_line_score(matrix_T, i, j)
                    if pts > 0:
                        score += pts
                        processed_cols.add(i)
    # print(score)
    return score



lines_horizontal=[]
for i in range(0,1601,100):
    l=[]
    l.append((0,i))
    l.append((1599,i))
    lines_horizontal.append(l)
    
lines_vertical=[]
for i in range(0,1601,100):
    l=[]
    l.append((i,0))
    l.append((i,1599))
    lines_vertical.append(l)

files=os.listdir(data_file_path)
files.sort()
for file in files:
    if file[-3:]=='jpg':
        img = cv.imread(data_file_path+file)
        result, result_padded=extrage_careu(img)
        if file[-6:-4]=='00':
            config_up_left, config_up_left_points=extract_initial_config(result[0:800,0:800])
            config_up_right, config_up_right_points=extract_initial_config(result[0:800,800:1600])
            config_down_left, config_down_left_points=extract_initial_config(result[800:1600,0:800])
            config_down_right, config_down_right_points=extract_initial_config(result[800:1600,800:1600])
            full_config_matrix=np.concatenate((np.concatenate((config_up_left,config_up_right),axis=1),np.concatenate((config_down_left,config_down_right),axis=1)),axis=0)
            full_points_matrix=np.concatenate((np.concatenate((config_up_left_points,config_up_right_points),axis=1),np.concatenate((config_down_left_points,config_down_right_points),axis=1)),axis=0)
            full_object_matrix=np.full(full_config_matrix.shape,"")
            # print(full_config_matrix)
            prev = result
        else:
            # show_image('previous',prev)
            # show_image('current',result)
            full_config_matrix_prev=full_config_matrix.copy()
            full_config_matrix=new_config_matrix(prev,result)
            # print(full_config_matrix)
            prev = result
        for i in range(16):
            for j in range(16):
                if full_config_matrix[i][j]==1:
                    template_img=extract_template(result_padded[i*100:(i+1)*100+50,j*100:(j+1)*100+50])
                    extracted_color=extract_color(result[i*100:(i+1)*100,j*100:(j+1)*100])
                    if file[-6:-4]!='00' and full_object_matrix[i][j] == "":
                        text_rezultat = f"{i+1}{chr(j+ord('A'))} {template_img+1}{extracted_color[0].upper()}"
                        # print(text_rezultat)

                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        nume_txt = file.replace('.jpg', '.txt')
                        cale_txt = os.path.join(output_dir, nume_txt)
                        with open(cale_txt, 'a') as f:
                            f.write(f"{text_rezultat}\n")
                            
                    full_object_matrix[i][j]=(f"{template_img}{extracted_color[0].upper()}")
        if file[-6:-4] != "00":
            score = calculate_score(full_config_matrix, full_config_matrix_prev, full_points_matrix)
            nume_txt = file.replace('.jpg', '.txt')
            cale_txt = os.path.join(output_dir, nume_txt)
            with open(cale_txt, 'a') as f:
                f.write(f"{score}\n")

        



