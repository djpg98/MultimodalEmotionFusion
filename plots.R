library(ggplot2)
library(tools)

args <- commandArgs(trailingOnly = TRUE)
task <- args[1]
file_name <- args[2]

csv_prefix <- file.path(task, "Training Data")
plot_prefix <- file.path(task, "Plots")

loss_file <- file.path(csv_prefix, paste(file_name, "_loss.csv", sep = ""))
acc_file <- file.path(csv_prefix, paste(file_name, "_acc.csv", sep = ""))

loss <- read.csv(loss_file, header = T)
acc <- read.csv(acc_file, header = T)

ggplot(loss) + geom_line(aes(x = epoch, y = train, color = "Training")) + geom_line(aes(x = epoch, y = val, color = "Validation")) + scale_color_discrete('Mode') + xlab('Epoch') + ylab('Loss') + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10)) # nolint
plot_name <- file.path(plot_prefix, paste(file_name, "_loss.png", sep = ""))
ggsave(plot_name)

ggplot(acc) + geom_line(aes(x = epoch, y = train, color = "Training")) + geom_line(aes(x = epoch, y = val, color = "Validation")) + scale_color_discrete('Mode') + xlab('Epoch') + ylab('Accuracy') + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10)) # nolint
plot_name <- file.path(plot_prefix, paste(file_name, "_acc.png", sep = ""))
ggsave(plot_name)
