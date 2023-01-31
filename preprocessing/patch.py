def my_patch(img, patch_size=128, overlap=0.2):
    """
    List of patches

    """
    import math

    h, w = img.shape

    step = int(math.ceil(patch_size * (1 - overlap)))

    x1 = 0
    y1 = 0
    x2, y2 = patch_size, patch_size
    patch_ls = []
    patch_ind = []
    break_col = False
    col_items = 0
    while True:  # col loop
        if y2 > h:
            y2 = h
            y1 = y2 - patch_size
            break_col = True
        break_row = False
        row_items = 0
        while True:  # row loop
            # if (x2 - w) >= step // 0.5:
            #     x2 = w
            #     x1 = x2 - patch_size
            #     break_row = True

            p = img[y1:y2, x1:x2]
            patch_ls.append(p)
            patch_ind.append((x1, x2, y1, y2))
            row_items += 1
            # if break_row:
            #     break_row = True
            #     x1 = 0
            #     x2 = patch_size
            #     break
            if not (-(x2 - w) >= step * overlap):
                x1 = 0
                x2 = patch_size
                break
            x1 += step
            x2 += step
        col_items += 1
        if break_col:
            break
        # if not (-(y2 - h) >= step * overlap):
        #     break
        y1 += step
        y2 += step
    return patch_ls, patch_ind, row_items, col_items
