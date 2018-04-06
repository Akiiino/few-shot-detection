import pandas as pd
import numpy as np
from collections import Counter
import os
from tqdm import tqdm, tqdm_pandas
from PIL import Image
from scipy import stats

ONLY_BG = False

np.random.seed(1337)

def loader(path, x, y, w, h):
    c_x = x + 0.5 * w
    c_y = y + 0.5 * h

    w_r = h_r = 2 ** int(np.ceil(np.log2(max(w, h))))

    if w_r % 2 != 0 or h_r % 2 != 0:
        print(w, h, w_r, h_r)
        raise ArithmeticError()

    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB').crop((
            c_x - 0.5 * w_r,
            c_y - 0.5 * h_r,
            c_x + 0.5 * w_r,
            c_y + 0.5 * h_r
        ))

        return img, (w_r, h_r)


def make_dataset(img_dir, gt, proc_dir):
    def get_size(dir_, path):
        with open(os.path.join(dir_, path), 'rb') as f:
            return Image.open(f).size


    def get_bg(row, w_loc, w_scale, h_loc, h_scale):
        # width = int(np.random.exponential(w_scale) + w_loc)
        # height = int(np.random.exponential(h_scale) + h_loc)
        width = row["width"]
        height = row["height"]

        x = np.random.randint(0, row["x_size"] - width)
        y = np.random.randint(0, row["y_size"] - height)
        return (row["filename"], x, y, width, height, "bg", -1, row["x_size"], row["y_size"])

    img_dir = os.path.expanduser(img_dir)

    sizes = gt.apply(lambda x: get_size(img_dir, x["filename"]), axis=1)
    gt = pd.concat([gt, pd.DataFrame(np.stack(sizes), columns=["x_size", "y_size"])], axis=1)

    w_params = stats.expon.fit(gt["width"])
    h_params = stats.expon.fit(gt["height"])

    bgs = pd.DataFrame(
        list(
            gt.sample(n=2*int(gt.groupby("sign_class").count().mean().mean())).apply(
                lambda x: get_bg(x, *w_params, *h_params), axis=1
            )
        ),
        columns=gt.columns
    )

    if ONLY_BG:
        gt = bgs.reset_index(drop=True)
    else:
        gt = gt.append(bgs).reset_index(drop=True)

    classes = {c: 0 for c in gt["sign_class"]}

    os.makedirs("proc_data", exist_ok=True)

    for class_ in classes:
        os.makedirs(os.path.join(proc_dir, class_), exist_ok=True)


    c = Counter()

    for i, fname, x, y, w, h, class_, *_ in tqdm(gt.itertuples(), total=len(gt)):
        path = os.path.join(img_dir, fname)

        img, shape = loader(path, x, y, w, h)

        with open(os.path.join(proc_dir, class_, str(classes[class_])) + ".png", "wb") as img_path:
            img.save(img_path)

        c.update([shape])

        classes[class_] += 1

    print(c.most_common())

make_dataset("data/rtsd-frames", pd.read_csv("data/full-gt.csv"), "proc_data")
