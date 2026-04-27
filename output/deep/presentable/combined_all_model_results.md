# SortSmart Combined Model Results

| Family | Model | Experiment | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Latency ms |
|---|---|---|---:|---:|---:|---:|
| Pretrained Deep | vit_base_patch16_224 | full_finetune_5ep_lr1e-4 | 0.9049 | 0.9219 | 0.9263 | 5.2277 |
| Pretrained Deep | swin_tiny_patch4_window7_224 | full_finetune_5ep_lr1e-4 | 0.9030 | 0.9475 | 0.9474 | 9.9769 |
| Pretrained Deep | convnext_tiny | full_finetune_5ep_lr1e-4 | 0.8929 | 0.9191 | 0.9316 | 6.1974 |
| Pretrained Deep | efficientnetv2_s | full_finetune_5ep_lr1e-4 | 0.7857 | 0.7824 | 0.7895 | 16.8763 |
| Classical ML | Gradient Boosting | HOG+HSV | 0.7847 | 0.7833 | 0.7895 | 7.4800 |
| Classical ML | Random Forest | HOG+HSV | 0.7323 |  |  |  |
| Custom CNN | custom_cnn_scratch | scratch_light_weighted_loss_70ep_ls0.05 | 0.7197 | 0.6961 | 0.7289 | 1.9565 |
| Classical ML | RBF SVM | HOG+HSV | 0.7060 |  |  |  |
| Custom CNN | custom_cnn_scratch | scratch_light_weighted_loss_35ep | 0.6672 | 0.6026 | 0.6342 | 1.9768 |
| Classical ML | Logistic Regression | HOG+HSV | 0.6340 |  |  |  |
| Pretrained Deep | convnext_tiny | head_only_5ep_lr1e-4 | 0.6336 | 0.6247 | 0.6711 | 5.6632 |
| Pretrained Deep | swin_tiny_patch4_window7_224 | head_only_5ep_lr1e-4 | 0.5903 | 0.6252 | 0.6553 | 9.4643 |
| Pretrained Deep | vit_base_patch16_224 | head_only_5ep_lr1e-4 | 0.5830 | 0.6423 | 0.6711 | 5.1868 |
| Classical ML | Linear SVM | HOG+HSV | 0.5707 |  |  |  |
| Custom CNN | custom_cnn_scratch | scratch_strong_weighted_loss_35ep | 0.5656 | 0.5104 | 0.5395 | 2.0109 |
| Custom CNN | custom_cnn_scratch | scratch_strong_weighted_sampler_35ep | 0.5206 | 0.4761 | 0.5105 | 2.6402 |
| Pretrained Deep | efficientnetv2_s | head_only_5ep_lr1e-4 | 0.1435 | 0.1638 | 0.1711 | 15.7980 |
