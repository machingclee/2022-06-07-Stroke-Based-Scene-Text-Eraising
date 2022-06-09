input_height = 128
input_width = 640

cropped_bg_dir = "training_data/cropped_background"
cropped_txt_dir = "training_data/cropped_text"
cropped_txt_mask_dir = "training_data/cropped_text_mask"

test_cropped_bg_dir = "testing_data/cropped_background"
test_cropped_txt_dir = "testing_data/cropped_text"
test_cropped_txt_mask_dir = "testing_data/cropped_text_mask"

pths_dir = "pths"


epoches = 10
batch_size = 8
start_epoch = 1
sample_result_per_n_images = 160
results_dir = "results"
bar_format = "{desc}: {percentage:.1f}%|{bar:15}| {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
beta = (0.9, 0.999)
