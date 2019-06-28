import numpy as np
import pandas as pd
import pandas.plotting as plotting
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats


def stars(p):
    #    if p < 0.0001:
    #        return "****"
    #    elif (p < 0.001):
    #        return "***"
    if (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return ""


def main():
    file_in = "E:/result/generalization/a1_gen.csv"
    file_out = "E:/result/hypothesis_test/new_vae_gen.png"
    df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
    # df2 = pd.read_csv("E:/result/hypothesis_test/Gen_test.csv", sep=',', na_values=".", header=None)
    df.columns = ['β-VAE', 'PCA']


    method_name = ["β-VAE", "PCA"]
    method_name2 = ["変分自己符号化器", "主成分分析"]

    fig = plt.figure(figsize=(7, 7))
    ax1 = plt.subplot()

    # ax2 = plt.subplot2grid((4,1), (3,0))

    label = []
    xlabel_name = []
    # for name in method_name:
    label.append(df["β-VAE"])
    label.append(df["PCA"])
    xlabel_name.append("β-VAE")
    xlabel_name.append("PCA")

    # for name in method_name2:
    xlabel_name.append("β-VAE")
    xlabel_name.append("PCA")

    bp = ax1.boxplot(label, patch_artist=True)

    for box in bp['boxes']:
        box.set(color="black", linewidth=1.5)
    for box in bp['medians']:
        plt.setp(box, color="black", linewidth=1.5)
    for box in bp['caps']:
        plt.setp(box, color="black", linewidth=1.5)
    for box in bp['whiskers']:
        plt.setp(box, ls="solid", color="black", linewidth=1.5)

    colors = []
    for index in enumerate(method_name):
        colors.append('lightblue')
        colors.append('pink')

    for box, color in zip(bp["boxes"], sns.color_palette(colors)):
        box.set_facecolor(color)

    const = 1.0

    y_max = np.max(np.concatenate((df['β-VAE'], df['PCA'])))

    z, p = stats.wilcoxon(df['β-VAE'], df['PCA'])
    p_value = p * 2
    if (p_value <= 0.05):
        if df['β-VAE'].median() >= df['PCA'].median():
            color2 = 'blue'
        else:
            color2 = 'hotpink'
        ax1.annotate("", xy=(const, y_max + 0.002), xycoords='data',
                     xytext=(const + 1, y_max + 0.002), textcoords='data',
                     arrowprops=dict(arrowstyle="-", ec="red",
                                     connectionstyle="bar,fraction=0.15"))
        ax1.text(const + 0.50, y_max + 0.01, stars(p_value),
                 horizontalalignment='center',
                 verticalalignment='center',
                 color=color2)

    # ax1.set_xticklabels(xlabel_name, fontdict = {"fontproperties": font}, rotation=90)
    ax1.set_xticklabels(xlabel_name, rotation=90, fontsize=14)
    ax1.set_ylabel("Generalization")
    # ax1.set_ylabel("Specificity")

    ax1.set_ylim([0, 0.08])
    ax1.grid(which="both")

    df_mean = pd.DataFrame()
    df_mini = pd.DataFrame()
    mean = np.zeros(2, dtype=np.float32)
    mini = np.zeros(2, dtype=np.float32)
    # for name, name2 in zip(method_name, method_name2):
    mean[0] = df['β-VAE'].mean()
    mean[1] = df['PCA'].mean()
    mini[0] = df['β-VAE'].min()
    mini[1] = df['PCA'].min()

    df_mean['β-VAE'] = np.array(mean)
    df_mini['β-VAE'] = np.array(mini)
    mean = df_mean.mean(axis='columns')
    df_mean['平均値'] = np.array(mean)
    df_mean = df_mean.round(4)
    mean2 = df_mini.mean(axis='columns')
    df_mini['平均値'] = np.array(mean2)
    df_mini = df_mini.round(4)

    df_st = pd.concat([df_mean, df_mini])

    # plotting.table(ax2, df_mean, loc='center')
    # ax2 = fig.add_axes([0.1, 0, 0.9, 0.2])
    # ax2.axis('off')
    # ax2 = plt.table(cellText=df_st.values, rowLabels=('平均値', '平均値', '最小値', '最小値'),
    #                 rowColours=('lightblue', 'pink', 'lightblue', 'pink'), colLabels=df_st.columns.values, loc='bottom',
    #                 cellLoc='center', rowLoc='center', bbox=[0, 0, 1.0, 0.9])
    # ax2.set_fontsize(20)
    #
    # plt.tight_layout()
    plt.show()
    # plt.savefig(file_out)
    # plt.savefig('\\\\tera\\share\\打ち合わせ報告書\\2018\\後期\\knmr\\20181224\\A_DICE_1129DSV333_Mvs1129DSV333_M2.png')
    # plt.savefig('C:\\Users\\Kanamori\\Documents\\デスクトップ\\images\\A_DICE_1129DSV333_M2vs1129DSV721_M2.png')

if __name__ == '__main__':
    main()