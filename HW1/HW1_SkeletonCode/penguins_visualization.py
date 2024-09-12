from cProfile import label

import matplotlib.container
from matplotlib import matplotlib_fname

try:
    import numpy as np
    import timeit
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import palmer
except ImportError:
    import pip
    pip.main(['install', 'seaborn', 'matplotlib', 'pandas', 'numpy'])
    import numpy as np
    import timeit
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import palmer


def question1():
    # Function to draw histogram
    def draw_species_histogram(data: pd.DataFrame, axis: int):
        ax = sns.histplot(data=data, x="species", hue="species", ax=axis, discrete=True)

        # Add bar labels
        for c in ax.containers:
            ax.bar_label(c)

    df = palmer.load_csv("data/penguins.csv")
    print(df.head())

    # Side by side for easier comparison
    fig, axs = plt.subplots(1, 2)

    # Title and Labels
    fig.suptitle("Palmer Penguins Species Distribution")
    fig.supxlabel("Species")
    axs[0].set_title("Original Data")
    axs[1].set_title("Post remove_na Data")

    # Draw histograms (Shouldn't we use bar plots since we are working with categorical data?)
    draw_species_histogram(df, axs[0])
    draw_species_histogram(palmer.remove_na(df, "species"), axs[1])

    # Clear individual subplot labels
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")

    # Cleanup and display
    plt.tight_layout()
    plt.show()


def question2():
    # Data loading and setup
    df = palmer.load_csv("data/penguins.csv")
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    numerical_feature_names = ["Bill length (mm)",
                               "Bill depth (mm)",
                               "Flipper length (mm)",
                               "Body mass (g)"]

    # Canvas setup
    fig, axs = plt.subplots(1, 4)
    fig.suptitle("Numeric Feature Boxplots per Species")
    fig.supxlabel("Species")

    plots = []
    labels = ["Adelie", "Gentoo", "Chinstrap"]

    # Plot each boxplot
    for i, feature in enumerate(numerical_features):
        plots.append(sns.boxplot(data=df, x="species", y=feature, ax=axs[i], hue="species"))
        axs[i].set_xticklabels([])
        axs[i].set_ylabel(numerical_feature_names[i])
        axs[i].set_xlabel("")

    # Legend - this legend took way too long to position properly. Will look for a better way to do this
    fig.legend(labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.075), ncol=3)
    plt.tight_layout(rect=(0, 0.1, 1, 1))

    plt.show()


def question3():
    pass

if __name__ == '__main__':
    # question1()
    # question2()
    question3()