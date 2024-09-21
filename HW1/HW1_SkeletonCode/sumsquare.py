try:
    import numpy as np
    import timeit
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    import pip
    pip.main(['install', 'seaborn', 'matplotlib', 'pandas', 'numpy'])
    import numpy as np
    import timeit
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

"""
Vectorization Comparison for Computing Sum of Squares
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""


def gen_random_samples(n):
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size n
        An array of n random samples
    """
    return np.random.randn(n)


def sum_squares_for(samples: np.array):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    
    ret = 0
    for s in samples:
        ret += s**2
    
    return ret


def sum_squares_np(samples: np.array):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    return samples @ samples


def time_ss(sample_list: list[int]):
    """
    Time it takes to compute the sum of squares
    for varying number of samples. The function should
    generate a random sample of length s (where s is an 
    element in sample_list), and then time the same random 
    sample using the for and numpy loops.

    Parameters
    ----------
    sample_list : list of length n
        A list of integers to .

    Returns
    -------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.
    """
    ss_dict = dict()
    ss_dict['n'] = []
    ss_dict['ssfor'] = []
    ss_dict['ssnp'] = []

    for n in sample_list:
        # Insert n
        data = gen_random_samples(n)
        ss_dict['n'].append(n)

        # Time ssfor
        start = timeit.default_timer()
        sum_squares_for(data)
        ss_dict['ssfor'].append(timeit.default_timer() - start)

        # Time ssnp
        start = timeit.default_timer()
        sum_squares_np(data)
        ss_dict['ssnp'].append(timeit.default_timer() - start)

    return ss_dict


def timess_to_df(ss_dict: dict):
    """
    Time the time it takes to compute the sum of squares
    for varying number of samples.

    Parameters
    ----------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.

    Returns
    -------
    time_df : Pandas dataframe that has n rows and 3 columns.
        The column names must be n, ssfor, ssnp and follow that order.
        ssfor and ssnp should contain the time in seconds.
    """

    ret = pd.DataFrame.from_dict(ss_dict)
    print(ret.head(None))
    return ret


def main():
    # generate 100 samples
    samples = gen_random_samples(100)
    # call the for version
    ss_for = sum_squares_for(samples)
    # call the numpy version
    ss_np = sum_squares_np(samples)
    # make sure they are approximately the same value
    import numpy.testing as npt
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)


def compare_times(problem_sizes: list[int]=[int(10 ** i) for i in np.linspace(1, 8, 15)]):
    """
    Generates and displays a matplotlib scatter plot using seaborn to compare the running time of ssfor and ssnp
    Read this section for Q3 part f

    Parameters
    ----------
    problem_sizes: list of problems sizes that the two versions of the ss algorithm will take on.

    Returns
    -------
    None
    """

    timings = timess_to_df(time_ss(problem_sizes))
    sns.scatterplot(data=timings, x='n', y='ssnp', marker='s')
    sns.scatterplot(data=timings, x='n', y='ssfor', marker='o')
    plt.xlabel("Problem size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.xscale('log') # Easier to see
    plt.legend(labels=['ssnp', 'ssfor'])
    plt.grid()
    plt.title("Runtime comparison for squaring n numbers:\nfor loop VS numpy")
    plt.show()

if __name__ == "__main__":
    main()
    compare_times()

