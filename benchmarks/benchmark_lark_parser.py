"""
Based on:
    https://github.com/Erotemic/misc/blob/main/tests/python/bench_template.py

Requirements:
    pip install ubelt timerit pandas numpy seaborn matplotlib
"""


def random_lark_grammar(size):
    """
    TODO: could likely be more sophisticated with how we *generate* random
    text. (Almost as if that's what CFGs do!).
    """
    lines = [
        'start: final',
        'simple_rule_0 : CNAME'
    ]
    idx = 0
    for idx in range(1, size):
        lines.append(f'simple_rule_{idx} : "(" simple_rule_{idx - 1} ")"')

    lines.append(f'final : simple_rule_{idx} "."')
    lines.append('%import common.CNAME')
    text = '\n'.join(lines)
    return text


def _autompl_lite():
    """
    A minimal port of :func:`kwplot.autompl`

    References:
        https://gitlab.kitware.com/computer-vision/kwplot/-/blob/main/kwplot/auto_backends.py#L98
    """
    import ubelt as ub
    import matplotlib as mpl
    interactive = False
    if ub.modname_to_modpath('PyQt5'):
        # Try to use PyQt Backend
        mpl.use('Qt5Agg')
        try:
            __IPYTHON__
        except NameError:
            pass
        else:
            import IPython
            ipython = IPython.get_ipython()
            ipython.magic('pylab qt5 --no-import-all')
            interactive = True
    return interactive


def benchmark():
    import ubelt as ub
    import pandas as pd
    import timerit
    import numpy as np
    import lark
    import lark_cython

    grammar_fpath = ub.Path(lark.__file__).parent / 'grammars/lark.lark'
    grammar_text = grammar_fpath.read_text()

    cython_parser = lark.Lark(grammar_text,  start='start', parser='lalr', _plugins=lark_cython.plugins)
    python_parser = lark.Lark(grammar_text,  start='start', parser='lalr')

    def parse_cython(text):
        cython_parser.parse(text)

    def parse_python(text):
        python_parser.parse(text)

    method_lut = locals()  # can populate this some other way

    # Change params here to modify number of trials
    ti = timerit.Timerit(300, bestof=10, verbose=1)

    # if True, record every trail run and show variance in seaborn
    # if False, use the standard timerit min/mean measures
    RECORD_ALL = True

    # These are the parameters that we benchmark over
    basis = {
        'method': [
            'parse_python',
            'parse_cython',
        ],
        'size': np.linspace(16, 512, 8).round().astype(int),
    }
    xlabel = 'size'
    # Set these to param labels that directly transfer to method kwargs
    kw_labels = []
    # Set these to empty lists if they are not used
    group_labels = {
        'style': [],
        'size': [],
    }
    group_labels['hue'] = list(
        (ub.oset(basis) - {xlabel}) - set.union(*map(set, group_labels.values())))
    grid_iter = list(ub.named_product(basis))

    # For each variation of your experiment, create a row.
    rows = []
    for params in grid_iter:
        group_keys = {}
        for gname, labels in group_labels.items():
            group_keys[gname + '_key'] = ub.repr2(
                ub.dict_isect(params, labels), compact=1, si=1)
        key = ub.repr2(params, compact=1, si=1)
        # Make any modifications you need to compute input kwargs for each
        # method here.
        kwargs = ub.dict_isect(params.copy(),  kw_labels)
        kwargs['text'] = random_lark_grammar(params['size'])
        method = method_lut[params['method']]
        # Timerit will run some user-specified number of loops.
        # and compute time stats with similar methodology to timeit
        for timer in ti.reset(key):
            # Put any setup logic you dont want to time here.
            # ...
            with timer:
                # Put the logic you want to time here
                method(**kwargs)
        if RECORD_ALL:
            # Seaborn will show the variance if this is enabled, otherwise
            # use the robust timerit mean / min times
            chunk_iter = ub.chunks(ti.times, ti.bestof)
            times = list(map(min, chunk_iter))
            for time in times:
                row = {
                    # 'mean': ti.mean(),
                    'time': time,
                    'key': key,
                    **group_keys,
                    **params,
                }
                rows.append(row)
        else:
            row = {
                'mean': ti.mean(),
                'min': ti.min(),
                'key': key,
                **group_keys,
                **params,
            }
            rows.append(row)

    time_key = 'time' if RECORD_ALL else 'min'

    # The rows define a long-form pandas data array.
    # Data in long-form makes it very easy to use seaborn.
    data = pd.DataFrame(rows)
    data = data.sort_values(time_key)
    print(data)

    if RECORD_ALL:
        # Show the min / mean if we record all
        min_times = data.groupby('key').min().rename({'time': 'min'}, axis=1)
        mean_times = data.groupby('key')[['time']].mean().rename({'time': 'mean'}, axis=1)
        stats_data = pd.concat([min_times, mean_times], axis=1)
        stats_data = stats_data.sort_values('min')
        print('Statistics:')
        print(stats_data)

    plot = True
    if plot:
        # import seaborn as sns
        # kwplot autosns works well for IPython and script execution.
        # not sure about notebooks.
        interactive = _autompl_lite()
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set()

        plotkw = {}
        for gname, labels in group_labels.items():
            if labels:
                plotkw[gname] = gname + '_key'

        # Your variables may change
        fig = plt.figure()
        fig.clf()
        ax = fig.gca()
        sns.lineplot(data=data, x=xlabel, y=time_key, marker='o', ax=ax, **plotkw)
        ax.set_title('Benchmark Python Grammar')
        ax.set_xlabel('Input Size')
        ax.set_ylabel('Time (seconds)')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        if not interactive:
            plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python benchmarks/benchmark_lark_parser.py
    """
    benchmark()
