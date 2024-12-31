import matplotlib.pyplot as plt
import numpy as np

# participants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# accuracy = [0.991, 0.979, 0.935, 0.997, 0.983, 0.983, 0.968, 0.973, 0.974, 0.981, 0.975, 0.972]
# f1_score = [0.818, 0.822, 0.776, 0.857, 0.728, 0.747, 0.630, 0.658, 0.763, 0.872, 0.781, 0.747]
# recall = [0.951, 0.882, 0.948, 0.851, 0.847, 0.642, 0.896, 0.568, 0.863, 0.866, 0.847, 0.793]
# precision = [0.717, 0.770, 0.657, 0.864, 0.638, 0.891, 0.485, 0.783, 0.684, 0.879, 0.724, 0.706]
#
# overall_accuracy = np.mean(accuracy)
# overall_f1_score = np.mean(f1_score)
# overall_recall = np.mean(recall)
# overall_precision = np.mean(precision)
#
# bar_width = 0.1
# x = np.arange(len(participants))
# bar_colors = ['black', 'deepskyblue', 'grey', 'royalblue']
# line_colors = ['black', 'deepskyblue', 'grey', 'royalblue']
#
# plt.figure(figsize=(16, 9))
#
# plt.bar(x - 1.5 * bar_width, accuracy, bar_width, label="Accuracy", color=bar_colors[0], alpha=0.7)
# plt.bar(x - 0.5 * bar_width, f1_score, bar_width, label="F1 Score", color=bar_colors[1], alpha=0.7)
# plt.bar(x + 0.5 * bar_width, recall, bar_width, label="Recall", color=bar_colors[2], alpha=0.7)
# plt.bar(x + 1.5 * bar_width, precision, bar_width, label="Precision", color=bar_colors[3], alpha=0.7)
#
# plt.axhline(overall_accuracy, color=line_colors[0], linestyle='--', linewidth=2, label="Avg Accuracy")
# plt.axhline(overall_f1_score, color=line_colors[1], linestyle='--', linewidth=2, label="Avg F1 Score")
# plt.axhline(overall_recall, color=line_colors[2], linestyle='--', linewidth=2, label="Avg Recall")
# plt.axhline(overall_precision, color=line_colors[3], linestyle='--', linewidth=2, label="Avg Precision")
#
# plt.text(len(participants) - 0.5, overall_accuracy + 0.01, f"{overall_accuracy:.3f}",
#          color=line_colors[0], fontsize=12, fontweight='bold', fontstyle='italic')
# plt.text(len(participants) - 0.5, overall_f1_score + 0.01, f"{overall_f1_score:.3f}",
#          color=line_colors[1], fontsize=12, fontweight='bold', fontstyle='italic')
# plt.text(len(participants) - 0.5, overall_recall + 0.01, f"{overall_recall:.3f}",
#          color=line_colors[2], fontsize=12, fontweight='bold', fontstyle='italic')
# plt.text(len(participants) - 0.5, overall_precision + 0.005, f"{overall_precision:.3f}",
#          color=line_colors[3], fontsize=12, fontweight='bold', fontstyle='italic')
#
# model_params = (
#     r"$\mathbf{LinearSVC(C=10000, dual='auto', loss='squared\_hinge', penalty='l2',}$" "\n"
#     r"$\mathbf{class\_weight=\{0: 2, 1: 6\}, max\_iter=1000)}$"
# )
# plt.text(
#     0.02, 0.02, model_params, fontsize=10, color="black",
#     transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
# )
#
# plt.title("Participant-wise metrics for LinearSVC", fontsize=16)
# plt.xlabel("Participants", fontsize=14)
# plt.ylabel("Score", fontsize=14)
# plt.xticks(x, participants, fontsize=12)
# plt.ylim(0.0, 1.1)
# plt.legend(fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
label_0_counts = [135167, 135288, 143344, 124642, 129232, 135945, 136392, 132844, 135373, 136325, 132742, 130436]
label_1_counts = [5389, 11936, 9787, 3187, 8337, 5288, 11182, 7440, 10889, 12011, 10726, 8553]

plt.figure(figsize=(12, 6))
x = np.arange(len(participants))
width = 0.35

plt.bar(x - width/2, label_0_counts, width, color='black', label='Label 0')
plt.bar(x + width/2, label_1_counts, width, color='royalblue', label='Label 1')

for i, (count_0, count_1) in enumerate(zip(label_0_counts, label_1_counts)):
    plt.text(x[i] - width/2, count_0 + 1000, str(count_0), ha='center', va='bottom', fontsize=8)
    plt.text(x[i] + width/2, count_1 + 1000, str(count_1), ha='center', va='bottom', fontsize=8)

plt.xticks(x, participants)
plt.xlabel('participants')
plt.ylabel('count')
plt.title('participant-wise class count')
plt.legend()

plt.tight_layout()
plt.show()
