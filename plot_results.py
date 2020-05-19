import matplotlib.pyplot as plt


def plot_results(measures, target, estimates, dates, title, path):
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(measures, 'k+', label='noisy measurements')
    plt.plot(estimates, 'b-', label='prediction')
    plt.plot(target, color='g', label='true value')
    plt.title(title, fontweight='bold')

    plt.xlabel('dates')
    x_label_count = int(len(dates) / 70)
    xi = list(range(x_label_count))
    xi = [i * 70 for i in xi]
    x_axes = [dates[i] for i in xi]
    plt.xticks(xi, x_axes)

    plt.ylabel('volume per hour')
    plt.legend()
    plt.savefig(path)
    plt.show()
