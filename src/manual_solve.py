#!/usr/bin/python
"""
NUI Galway CT5132 Programming and Tools for AI - Assignment 3

By writing our names below and submitting this file, we declare that
all additions to the provided manual_solve.py skeleton file are our own work,
and that we have not seen any work on this assignment by another student/group.

Student name(s): Kalyani Prashant Kawale, Apoorva Patil
Student ID(s): 21237189, 21237465

Link to GitHub Repo: https://github.com/Kalyani011/ARC
"""
import os, sys
import json
import numpy as np
import re


def get_consecutive_num(arr):
    """
    Method to get indices of second number in a pair of consecutive numbers
    Note: solve_90f3ed37 uses this function
    """
    rows = []
    for i in range(len(arr) - 1):
        if (arr[i] + 1) == arr[i + 1]:
            rows.append(arr[i + 1])
    return rows


def rm_elements(arr, remove_rows):
    """
    Method to remove elements given in a list from an numpy array
    Note: solve_90f3ed37 uses this function
    """
    for element in remove_rows:
        arr = arr[arr != element]
    return arr


def solve_90f3ed37(x):
    """
    This method solves the task given in 90f3ed37.json file.
    Difficulty Level: Difficult
    Task Description:
    Given a grid of size 15x10 (All train and test samples had the same size),
    with cells colored in either black (0), blue (1) or cyan (8) colors,
    identify the pattern created by cyan colored cells across one or multiple rows,
    and apply the pattern to partially cyan color filled cells appearing below the pattern.
    The rows with the pattern to be applied are always above all incompletely colored cells.
    All the colored rows are separated by one or more rows of all black cells.

    # RESULT: This method solves all 4 train (3) and test (1) grids successfully
    """
    # getting indices of all rows consisting of only black(0) cells
    zero_indices = np.array([i for i in range(len(x)) if list(x[i, :]) == [0] * len(x[i, :])])
    # removing consecutive all black rows from zero_indices to maintain a single line of separation
    zero_indices = rm_elements(zero_indices, get_consecutive_num(zero_indices))

    # fetching the solution pattern for the grid
    pattern = x[zero_indices[0] + 1: zero_indices[1], :]

    # removing all zero(black) rows if any were fetched in the pattern
    indices = np.array([i for i in range(len(pattern)) if np.all(pattern[i] == 0, axis=0)])
    pattern = np.delete(pattern, indices.astype(int), axis=0)
    pattern_size = pattern.shape[0]

    # fetching all the rows with one or more colored cells from the grid
    rows_to_transform = [i for i in range(len(x)) if list(x[i, :]) != [0] * len(x[i, :])]
    # removing the rows containing the solution pattern from rows to be transformed
    to_transform = np.array(rows_to_transform[pattern_size:])
    # removing consecutive rows from to_transform to only contain starting row for drawing patterns
    to_transform = rm_elements(to_transform, get_consecutive_num(to_transform))

    # for each starting row in to_transform complete the pattern
    for start in to_transform:
        # setting the index of row where pattern ends
        end = start + pattern_size
        # getting rows where pattern is to be drawn
        chunk = np.copy(x[start:end, :])
        # shift flag checks if cells need to be shifted to fit pattern correctly
        shift = False
        # setting the shift flag to true if
        # for any row in pattern the corresponding row to be filled has more colored items
        # example: training first sample and test sample
        for i in range(len(pattern)):
            num_colored_pattern = len(pattern[i][pattern[i] != 0])
            num_colored_x = len(chunk[i][chunk[i] != 0])
            if num_colored_x > num_colored_pattern:
                shift = True
                break
        # getting the cells to be colored blue by subtracting sub-grid from pattern
        # if cells are to be shifted, processing the chunks to fit the pattern
        # and reconstruct the sub-grid
        if shift:
            chunk = np.delete(chunk, 0, 1)  # removing the first column initially
            col = np.zeros(chunk.shape[0])
            chunk = np.concatenate([chunk, col.reshape(-1, 1)], axis=1).astype(int)
            chunk = pattern - chunk  # fitting the pattern
            chunk = np.delete(chunk, -1, axis=1)
            # re - adding the first column
            chunk = np.concatenate((np.zeros(len(chunk)).astype(int).reshape(-1, 1), chunk), axis=1)
        else:
            chunk = pattern - chunk  # fitting the pattern
        # setting the color of newly added colored items in incomplete patterns to blue (1)
        for index in np.argwhere(chunk == 8):
            x[start:end, :][index[0], index[1]] = 1
    return x


def solve_7df24a62(x):
    pattern_list = []
    row, col = np.where(x == 1)
    min_row = np.min(row)
    min_col = np.min(col)
    max_row = np.max(row)
    max_col = np.max(col)
    pattern = x[min_row:max_row + 1, min_col:max_col + 1]
    i, j = np.where(pattern == 4)
    pat = pattern[np.min(i):np.max(i) + 1, np.min(j):np.max(j) + 1]
    copy_pat = np.copy(pat)
    # converting all 1s to 0s in the pattern
    copy_pat[copy_pat == 1] = 0
    pattern_shape = pat.shape
    pattern_list.append(copy_pat)
    pattern_list.append(np.transpose(copy_pat))
    pattern_list.append(np.flip(copy_pat))
    pattern_list.append(np.flip(copy_pat).T)
    pattern_list.append(np.fliplr(copy_pat))
    pattern_list.append(np.fliplr(copy_pat).T)
    pattern_list.append(np.flipud(copy_pat))
    pattern_list.append(np.flipud(copy_pat).T)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            subset_0 = x[row:row + pattern_shape[0], col:col + pattern_shape[1]]
            subset_1 = x[row:row + pattern_shape[1], col:col + pattern_shape[0]]
            for pattern in pattern_list:
                if subset_0.shape == pattern_shape and np.array_equal(subset_0, pattern):
                    subset_0[subset_0 != 4] = 1
                    x_start = row - 1
                    y_start = col - 1
                    x_end = row + pattern_shape[0]
                    y_end = col + pattern_shape[1]
                    x[x_start:x_end, y_start] = 1
                    x[x_start:x_end, y_end] = 1
                    if x_start != 0:
                        x[x_start, y_start:y_end] = 1
                    if y_end != x.shape[0] + 1:
                        y_end += 1
                    if x_end != len(x):
                        x[x_end, y_start:y_end] = 1

                if subset_1.shape[0] == pattern_shape[1] and subset_1.shape[1] == pattern_shape[0] and np.array_equal(
                        subset_1, pattern):
                    subset_1[subset_1 != 4] = 1
                    x_start = row - 1
                    y_start = col - 1
                    x_end = row + pattern_shape[1]
                    y_end = col + pattern_shape[0]
                    if x_start == -1:
                        x_start += 1
                    x[x_start:x_end, y_start] = 1
                    x[x_start:x_end, y_end] = 1
                    if x_start != 0:
                        x[x_start, y_start:y_end] = 1
                    if y_end != x.shape[0] + 1:
                        y_end += 1
                    if x_end != len(x):
                        x[x_end, y_start:y_end] = 1
    return x


def solve_a740d043(x):
    """
    This method solves the task given in a740d043.json file.
    Difficulty Level: Medium
    Task Description:
    Given a grid of any size m x n, with a background of blue (1) colored cells,
    and few cells colored in multiple colors, fetch a sub-grid consisting
    the rows and columns with colored cells (other than blue) and paint
    any blue cells in this sub-grid to black.

    # RESULT: This method solves all 4 train (3) and test (1) grids successfully
    """
    # getting all the rows and columns with cells that are not colored blue (1)
    rows, columns = np.where(x != 1)
    # getting rows with colored cells, with all columns
    row_x = x[list(set(rows))]
    # getting columns with colored cells from row_x
    x = row_x[:, min(columns):(max(columns) + 1)]
    # changing blue cells to black
    x[x == 1] = 0
    return x


def solve_8d510a79(x):
    """
    This method solves the task given in 8d510a79.json file.
    Difficulty Level: Medium
    Task Description:
    Given a grid of any size m x n, with a background of black (0) colored cells,
    and multiple cells colored in either blue (1), red (2) or gray (5),
    where gray cells form a boundary that divides the grid into two parts,
    for upper half of the grid of any shape,
    1. paint all rows above any blue colored cells in blue until we reach the top of grid for this cell's column
    2. paint all rows below any red colored cells in red until we reach the boundary for this cell's column
    for lower half of the grid of any shape,
    1. paint all rows below any blue colored cells in blue until we reach the end of the grid for this cell's column
    2. paint all rows above any red colored cells in red until we reach the boundary for this cell's column

    # RESULT: This method solves all 3 train (2) and test (1) grids successfully
    """
    # initializing blue and red with their color codes
    blue = 1
    red = 2
    # fetching all rows and columns with gray cells
    row, column = np.where(x == 5)
    # getting the indices of the boundary
    boundary = row[0]
    # initializing indices list for blue and red cells in upper and lower section of grid
    u_blue_indices = []
    u_red_indices = []
    l_blue_indices = []
    l_red_indices = []
    # separating the blue points found in grid into upper and lower lists
    for index in np.argwhere(x == blue):
        if index[0] < boundary:
            u_blue_indices.append(index)
        else:
            l_blue_indices.append(index)
    # separating the red points found in grid into upper and lower lists
    for index in np.argwhere(x == red):
        if index[0] < boundary:
            u_red_indices.append(index)
        else:
            l_red_indices.append(index)
    # coloring columns based on color and grid section
    for indices in u_blue_indices:
        x[:indices[0], indices[1]] = blue
    for indices in u_red_indices:
        x[indices[0]:boundary, indices[1]] = red
    for indices in l_blue_indices:
        x[indices[0]:, indices[1]] = blue
    for indices in l_red_indices:
        x[boundary + 1:indices[0], indices[1]] = red
    return x


def solve_94f9d214(x):
    """
    This method solves the task given in 94f9d214.json file.
    Difficulty Level: Easy to Medium
    Task Description:
    Given a grid of any size m x n, where m = 2n, with a background of black (0) colored cells,
    and multiple cells colored in either blue (1), or green (3), create a sub-grid of size n,
    where cells where blue and green cells overlap are colored black (0) and cells with no overlap,
    are colored red (2)
    # RESULT: This method solves all 5 train (4) and test (1) grids successfully
    """
    dimension = int(len(x) / 2)  # getting dimension for result grid
    red = 2  # initializing red with its color value
    # dividing input grid into two parts
    x_upper = x[:dimension, :]
    x_lower = x[dimension:, :]
    # converting green (3) cells to blue (1) for easier comparison
    x_upper[x_upper == 3] = 1
    # taking or of two sub grids to identify cells to be colored red
    result = np.logical_or(x_upper, x_lower)
    # getting indices for red cells
    red_x = np.argwhere(result == False)
    # creating the sub-grid with all black cells (0)
    x = np.zeros((dimension, dimension)).astype(int)
    # painting cells red(2) where there was no overlap of green(3) and blue(1) cells
    for index in red_x:
        x[index[0], index[1]] = red
    return x


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)

def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)

def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()
