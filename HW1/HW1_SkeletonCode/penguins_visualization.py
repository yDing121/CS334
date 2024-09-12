from matplotlib.lines import Line2D

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
        ax = sns.histplot(data=data, x="species", hue="species", ax=axis, discrete=True, palette='Set1') # Specify palette
        ax.legend_.remove() # Remove subplot legend

        # Add bar labels
        for c in ax.containers:
            ax.bar_label(c)

    # Load data
    df = palmer.load_csv("data/penguins.csv")
    print(df.head())

    # Side by side for easier comparison
    fig, axs = plt.subplots(1, 2, figsize=(9,6))

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

    # Manually create legend handles and labels - inspired by stackoverflow
    # Again - this legend took way too much effort to make (I came back to fix this legend after part 2 and part 3)
    palette = sns.color_palette('Set1', n_colors=3)
    labels = ["Adelie", "Gentoo", "Chinstrap"]
    # Create legend icons with a white line, circular marker, color corresponding to a label
    handles = [Line2D([0], [0], color='w', marker='o', markerfacecolor=palette[i]) for i in range(3)]

    # Create a single legend for the entire figure
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3)

    # Cleanup and display
    plt.tight_layout(rect=(0, 0.05, 1, 1))
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
    fig, axs = plt.subplots(1, 4, figsize=(9,6))
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
    df = palmer.load_csv("data/penguins.csv")
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    numerical_feature_names = ["Bill length (mm)",
                               "Bill depth (mm)",
                               "Flipper length (mm)",
                               "Body mass (g)"]
    fig, axs = plt.subplots(2, 3, figsize=(9,6))
    plots = []
    idx = 0

    for f1 in range(4):
        for f2 in range(f1+1, 4):
            cur_ax = axs[idx//3][idx%3]
            plots.append(sns.scatterplot(data=df, x=numerical_features[f1], y=numerical_features[f2], hue="species",
                                         ax=cur_ax))
            cur_ax.set_xlabel(numerical_feature_names[f1])
            cur_ax.set_ylabel(numerical_feature_names[f2])
            cur_ax.legend_.remove()
            idx += 1

    # Steal information from one of the plots and use it create a legend (since the labels are the same across plots)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3)

    fig.suptitle("Numeric Feature Boxplots per Species")
    plt.tight_layout(rect=(0, 0.1, 1, 1))

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    question1()
    question2()
    question3()
