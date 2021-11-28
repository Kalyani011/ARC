# The Abstraction and Reasoning Corpus (ARC) - Handcoded Solutions for 7 Tasks

This repository contains the ARC task data, as well as a browser-based interface for humans to try their hand at solving the tasks manually. Apart from the data and the web interface the repository also consists of a manual_solve.py python script that runs the hand-coded solutions for 7 tasks from data/training.

*"ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."*

A complete description of the dataset, its goals, and its underlying logic, can be found in: [The Measure of Intelligence](https://arxiv.org/abs/1911.01547).

As a reminder, a test-taker is said to solve a task when, upon seeing the task for the first time, they are able to produce the correct output grid for *all* test inputs in the task (this includes picking the dimensions of the output grid). For each test input, the test-taker is allowed 3 trials (this holds for all test-takers, either humans or AI).

## Description of tasks solved in manual_solve.py

**1. Task ID 90f3ed37**:
<img width="960" alt="90f3ed37" src="https://user-images.githubusercontent.com/43260930/143782491-975fbade-7b56-459e-a118-ec6504376902.PNG">
    
Difficulty Level: Difficult
- Given a grid of size 15x10 (All train and test samples had the same size), with cells colored in either black (0), blue (1) or cyan (8) colors, identify the pattern     created by cyan colored cells across one or multiple rows, and apply the pattern to partially cyan color filled cells appearing below the pattern.
- The rows with the pattern to be applied are always above all incompletely colored cells.
- All the colored rows are separated by one or more rows of all black cells.
      
**2. Task ID 7df24a62**:
<img width="960" alt="7df24a62" src="https://user-images.githubusercontent.com/43260930/143782569-ad469d3c-ef6f-4279-9576-3f07643c0159.PNG">
Difficulty Level: Difficult
- Given a grid of size 23x23 (All train and test samples had the same size), with cells colored in either black (0), blue (1) or yellow (4) colors, identify the pattern created by the yellow colored cells across multiple rows, and fill the rows and columns above and below the pattern grid (and it's all possible combinations) with blue color.
- The cells in the pattern grid that are not yellow also need to be filled with blue.
  
       
**3. Task ID a740d043**:
<img width="960" alt="a740d043" src="https://user-images.githubusercontent.com/43260930/143782634-a840a95c-1171-496a-8f61-56edd76016a2.PNG">
Difficulty Level: Medium    
- Given a grid of any size m x n, with a background of blue (1) colored cells, and few cells colored in multiple colors, fetch a sub-grid consisting the rows and columns with colored cells (other than blue) and paint any blue cells in this sub-grid to black.

**4. Task ID 8d510a79**:
<img width="959" alt="8d510a79" src="https://user-images.githubusercontent.com/43260930/143782690-29940077-632b-4548-a52a-a3a9fc5c5d83.PNG">
Difficulty Level: Medium
- Given a grid of any size m x n, with a background of black (0) colored cells, and multiple cells colored in either blue (1), red (2) or gray (5), where gray cells form a boundary that divides the grid into two parts.
- For upper half of the grid of any shape,
    1. paint all rows above any blue colored cells in blue until we reach the top of grid for this cell's column
    2. paint all rows below any red colored cells in red until we reach the boundary for this cell's column
- For lower half of the grid of any shape,
    1. paint all rows below any blue colored cells in blue until we reach the end of the grid for this cell's column
    2. paint all rows above any red colored cells in red until we reach the boundary for this cell's column

**5. Task ID 94f9d214**:
<img width="957" alt="94f9d214" src="https://user-images.githubusercontent.com/43260930/143782763-e1da5db4-3660-4745-be31-bc98c91250c4.PNG">
Difficulty Level: Easy to Medium
- Given a grid of any size m x n, where m = 2n, with a background of black (0) colored cells, and multiple cells colored in either blue (1), or green (3), create a sub-grid of size n, where cells where blue and green cells overlap are colored black (0) and cells with no overlap are colored red (2)
    
**6. Task ID d06dbe63**:
<img width="960" alt="d06dbe63" src="https://user-images.githubusercontent.com/43260930/143782896-0e7ef73c-f59e-4553-a70d-392e3fb69048.PNG">
Difficulty Level: Medium
- Given a grid of size 13x13 (All train and test samples had the same size), which contains one cell colored in cyan (8) and the remaining cells colored in black (0),
fill the cells above the cyan cell with color grey (5) in a step like fashion with two steps and moving towards the right.
- Similarly, for the cells below the cyan cell, the direction needs to be towards the left.
    
**7. Task ID 0ca9ddb6**:
<img width="957" alt="0ca9ddb6" src="https://user-images.githubusercontent.com/43260930/143782954-68318b5d-2659-4125-b262-242b999f5460.PNG">    
Difficulty Level: Easy
- Given a grid of size 9x9 (All train and test samples had the same size), which contains single cells colored in red (2), blue (1) and the remaining cells colored in black (0), fill the cells one-step diagonal to the red cells with yellow (4) and the cells one-step above and below the blue cell with orange (7).    
    
## Task file format

The `data` directory contains two subdirectories:

- `data/training`: contains the task files for training (400 tasks). Use these to prototype your algorithm or to train your algorithm to acquire ARC-relevant cognitive priors.
- `data/evaluation`: contains the task files for evaluation (400 tasks). Use these to evaluate your final algorithm. To ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its evaluation score as feedback).

The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
- `"test"`: test input/output pairs. It is a list of "pairs" (typically 1 pair).

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

When looking at a task, a test-taker has access to inputs & outputs of the demonstration pairs, plus the input(s) of the test pair(s). The goal is to construct the output grid(s) corresponding to the test input grid(s), using 3 trials for each test input. "Constructing the output grid" involves picking the height and width of the output grid, then filling each cell in the grid with a symbol (integer between 0 and 9, which are visualized as colors). Only *exact* solutions (all cells match the expected answer) can be said to be correct.


## Usage of the testing interface

The testing interface is located at `apps/testing_interface.html`. Open it in a web browser (Chrome recommended). It will prompt you to select a task JSON file.

After loading a task, you will enter the test space, which looks like this:

![test space](https://arc-benchmark.s3.amazonaws.com/figs/arc_test_space.png)

On the left, you will see the input/output pairs demonstrating the nature of the task. In the middle, you will see the current test input grid. On the right, you will see the controls you can use to construct the corresponding output grid.

You have access to the following tools:

### Grid controls

- Resize: input a grid size (e.g. "10x20" or "4x4") and click "Resize". This preserves existing grid content (in the top left corner).
- Copy from input: copy the input grid to the output grid. This is useful for tasks where the output consists of some modification of the input.
- Reset grid: fill the grid with 0s.

### Symbol controls

- Edit: select a color (symbol) from the color picking bar, then click on a cell to set its color.
- Select: click and drag on either the output grid or the input grid to select cells.
    - After selecting cells on the output grid, you can select a color from the color picking to set the color of the selected cells. This is useful to draw solid rectangles or lines.
    - After selecting cells on either the input grid or the output grid, you can press C to copy their content. After copying, you can select a cell on the output grid and press "V" to paste the copied content. You should select the cell in the top left corner of the zone you want to paste into.
- Floodfill: click on a cell from the output grid to color all connected cells to the selected color. "Connected cells" are contiguous cells with the same color.

### Answer validation

When your output grid is ready, click the green "Submit!" button to check your answer. We do not enforce the 3-trials rule.

After you've obtained the correct answer for the current test input grid, you can switch to the next test input grid for the task using the "Next test input" button (if there is any available; most tasks only have one test input).

When you're done with a task, use the "load task" button to open a new task.
