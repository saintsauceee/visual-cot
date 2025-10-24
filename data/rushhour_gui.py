import tkinter as tk
N = 6
CELL_SIZE = 60
GRID_SIZE = N*CELL_SIZE

def draw_grid(canvas: tk.Canvas, n: int, cell: int):
    """Draws an n x n grid of cell-sized squares on the given canvas."""
    for i in range(n + 1):
        y = i * cell
        canvas.create_line(0, y, n * cell, y, fill="black")
    for j in range(n + 1):
        x = j * cell
        canvas.create_line(x, 0, x, n * cell, fill="black")

def add_buttons(root):
    """Add all control buttons under the grid."""
    btn_frame = tk.Frame(root)
    btn_frame.pack(side = "left")

    tk.Button(btn_frame, text="Reset", width=14).grid(row=0)
    tk.Button(btn_frame, text="Red_Car", width=14).grid(row=1)
    tk.Button(btn_frame, text="Car", width=14).grid(row=2)
    tk.Button(btn_frame, text="Truck", width=14).grid(row=3)
    tk.Button(btn_frame, text="Rotate", width=14).grid(row=4)
    tk.Button(btn_frame, text="Textualize_Board", width=14).grid(row=5)
    tk.Button(btn_frame, text="Textualize_Move", width=14).grid(row=6)
    tk.Button(btn_frame, text="Save_To_File", width=14).grid(row=7)
    

if __name__ == "__main__":
    root = tk.Tk()
    root.title("RushHour Puzzle")
    add_buttons(root)
    canvas = tk.Canvas(root, width=GRID_SIZE, height=GRID_SIZE, bg="white")
    canvas.pack(side = "right")
    draw_grid(canvas, N, CELL_SIZE)
    

    root.mainloop()