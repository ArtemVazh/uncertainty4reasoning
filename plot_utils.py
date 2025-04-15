import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def get_cmap():
    cmap = matplotlib.cm.get_cmap('Greens')
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = 0.5
    return colors.ListedColormap(my_cmap)


def b_g(s, cmap, low=0, high=0):
    values = s
    if any(s.name.startswith(x) for x in ['ece', 'bs', 'rpp']):  # lower is better
        values = -values
    if isinstance(values.max(), str):
        return ['' for _ in values]
    rng = values.max() - values.min()
    norm = colors.Normalize(values.min() - (rng * low), values.max() + (rng * high))
    normed = norm(values.values)
    back_colors = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    text_colors = ["white" if x > 0.3 else "black" for x in normed]
    return [f'color: {text_color}; background-color: {color}' for text_color, color in zip(text_colors, back_colors)]


def pretty_plot_table(df, title=''):
    return df.style.apply(b_g, cmap=get_cmap()).set_caption(title).set_table_styles([
        {
            'selector': 'caption',
            'props': [
                ('caption-side', 'top'), ('text-align', 'right'),
                ('color', 'black'),
                ('font-size', '15px'),
                ('font-weight', 'bold')
            ]
        },
        {"selector": "th.row_heading", "props": [("text-align", "left")]},
        {"selector": "thead th", "props": [("border-bottom", "2px solid black")]},
        {"selector": "th.row_heading.level0", "props": [("border-top", "2px solid black")]}
    ])