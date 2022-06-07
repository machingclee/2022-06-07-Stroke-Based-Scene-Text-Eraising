input_height = 128
input_width = 640

bg_dir = "training_data/cropped_background"
txt_dir = "training_data/cropped_text"
txt_mask_dir = "training_data/text_mask"


epoches = 10
batch_size = 8
start_epoch = 1
sample_result_per_n_images = 160
results_dir = "results"
bar_format = "{desc}: {percentage:.1f}%|{bar:15}| {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
beta = (0.9, 0.999)
