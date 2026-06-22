import numpy as np


from config import ARENA_COORDS



def all_arena_pixels():

    arena_x = []
    arena_y = []
    for (x, y) in ARENA_COORDS:
        arena_x.append(x)
        arena_y.append(y)

    all_coords = []

    for i in range(min(arena_y), max(arena_y)):
        for j in range(min(arena_x), max(arena_x)):
            c = (j, i)
            all_coords.append(c)

    return all_coords

def pixel_exploration_score(x, y, exp_duration_frames):
    
    arena_pixels = all_arena_pixels
    num_px = len(all_arena_pixels)

    exp_score = np.full(len(exp_duration_frames), np.nan, dtype=float)

    for idx, i in enumerate(zip(x, y)):
        if i in arena_pixels:
            arena_pixels.remove(i)
        
        current_score = len(arena_pixels/num_px)
        exp_score[idx] = current_score

    return exp_score




