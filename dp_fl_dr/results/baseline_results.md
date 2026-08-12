# Step 1 Baseline Results

This baseline training compares basic centralized models without DP-FL on the IDRiD dataset.

## Test Accuracy Comparison

| Model | Mode | Test Accuracy | Target (Paper) | Difference |
|---|---|---|---|---|
| alexnet | 5-class | 46.60% | 74.09% | -27.49% |
| resnet | 5-class | 48.54% | 74.09% | -25.55% |
| squeezenet | 5-class | 45.63% | 74.09% | -28.46% |
| vgg | 5-class | 49.51% | 74.09% | -24.58% |
| alexnet | binary | 70.87% | 83.05% | -12.18% |
| resnet | binary | 81.55% | 83.05% | -1.50% |
| squeezenet | binary | 79.61% | 83.05% | -3.44% |
| vgg | binary | 76.70% | 83.05% | -6.35% |

## Training Logs (Binary Mode)

```text
Val Loss: 0.3345 Acc: 0.8065

Early stopping triggered
Best val Acc: 0.870968

--- Test Results for alexnet (binary) ---
Accuracy:  0.7087
Precision: 0.7294
Recall:    0.8986
F1 Score:  0.8052
Confusion Matrix:
[[11 23]
 [ 7 62]]


Training resnet for binary...
Epoch 1/100
----------
Train Loss: 0.6398 Acc: 0.7066
Val Loss: 0.6659 Acc: 0.7097

Epoch 2/100
----------
Train Loss: 0.5207 Acc: 0.7721
Val Loss: 0.5009 Acc: 0.8387

Epoch 3/100
----------
Train Loss: 0.4164 Acc: 0.7977
Val Loss: 0.4615 Acc: 0.8387

Epoch 4/100
----------
Train Loss: 0.3243 Acc: 0.8575
Val Loss: 0.4100 Acc: 0.8548

Epoch 5/100
----------
Train Loss: 0.2813 Acc: 0.8689
Val Loss: 0.3711 Acc: 0.8226

Epoch 6/100
----------
Train Loss: 0.2157 Acc: 0.9031
Val Loss: 0.3238 Acc: 0.8548

Epoch 7/100
----------
Train Loss: 0.2060 Acc: 0.9060
Val Loss: 0.3432 Acc: 0.8387

Epoch 8/100
----------
Train Loss: 0.1682 Acc: 0.9202
Val Loss: 0.3223 Acc: 0.7903

Epoch 9/100
----------
Train Loss: 0.1832 Acc: 0.9088
Val Loss: 0.2934 Acc: 0.8710

Epoch 10/100
----------
Train Loss: 0.1716 Acc: 0.9231
Val Loss: 0.2840 Acc: 0.8548

Epoch 11/100
----------
Train Loss: 0.1130 Acc: 0.9430
Val Loss: 0.2841 Acc: 0.8710

Epoch 12/100
----------
Train Loss: 0.0773 Acc: 0.9630
Val Loss: 0.3544 Acc: 0.8548

Epoch 13/100
----------
Train Loss: 0.0985 Acc: 0.9573
Val Loss: 0.3326 Acc: 0.8548

Epoch 14/100
----------
Train Loss: 0.0916 Acc: 0.9516
Val Loss: 0.3575 Acc: 0.8710

Epoch 15/100
----------
Train Loss: 0.0910 Acc: 0.9630
Val Loss: 0.4957 Acc: 0.8548

Epoch 16/100
----------
Train Loss: 0.0919 Acc: 0.9487
Val Loss: 0.3381 Acc: 0.8710

Epoch 17/100
----------
Train Loss: 0.0679 Acc: 0.9744
Val Loss: 0.3200 Acc: 0.8871

Epoch 18/100
----------
Train Loss: 0.0741 Acc: 0.9744
Val Loss: 0.3079 Acc: 0.8871

Epoch 19/100
----------
Train Loss: 0.0664 Acc: 0.9744
Val Loss: 0.3023 Acc: 0.8871

Epoch 20/100
----------
Train Loss: 0.0491 Acc: 0.9772
Val Loss: 0.3098 Acc: 0.8871

Epoch 21/100
----------
Train Loss: 0.0704 Acc: 0.9658
Val Loss: 0.3168 Acc: 0.8871

Epoch 22/100
----------
Train Loss: 0.0540 Acc: 0.9772
Val Loss: 0.3089 Acc: 0.8710

Epoch 23/100
----------
Train Loss: 0.0548 Acc: 0.9715
Val Loss: 0.3038 Acc: 0.8710

Epoch 24/100
----------
Train Loss: 0.0650 Acc: 0.9744
Val Loss: 0.2970 Acc: 0.8548

Epoch 25/100
----------
Train Loss: 0.0337 Acc: 0.9886
Val Loss: 0.3066 Acc: 0.8548

Epoch 26/100
----------
Train Loss: 0.0583 Acc: 0.9687
Val Loss: 0.3040 Acc: 0.8548

Epoch 27/100
----------
Train Loss: 0.0595 Acc: 0.9772
Val Loss: 0.3054 Acc: 0.8710

Epoch 28/100
----------
Train Loss: 0.0446 Acc: 0.9858
Val Loss: 0.3054 Acc: 0.8548

Epoch 29/100
----------
Train Loss: 0.0476 Acc: 0.9801
Val Loss: 0.3018 Acc: 0.8710

Epoch 30/100
----------
Train Loss: 0.0210 Acc: 0.9915
Val Loss: 0.3061 Acc: 0.8710

Epoch 31/100
----------
Train Loss: 0.0759 Acc: 0.9801
Val Loss: 0.3097 Acc: 0.8548

Epoch 32/100
----------
Train Loss: 0.0609 Acc: 0.9687
Val Loss: 0.3155 Acc: 0.8548

Early stopping triggered
Best val Acc: 0.887097

--- Test Results for resnet (binary) ---
Accuracy:  0.8155
Precision: 0.9032
Recall:    0.8116
F1 Score:  0.8550
Confusion Matrix:
[[28  6]
 [13 56]]


Training squeezenet for binary...
Epoch 1/100
----------
Train Loss: 0.6244 Acc: 0.6524
Val Loss: 0.5386 Acc: 0.8065

Epoch 2/100
----------
Train Loss: 0.5143 Acc: 0.7949
Val Loss: 0.4277 Acc: 0.8548

Epoch 3/100
----------
Train Loss: 0.4977 Acc: 0.7265
Val Loss: 0.5149 Acc: 0.7581

Epoch 4/100
----------
Train Loss: 0.4640 Acc: 0.7835
Val Loss: 0.3768 Acc: 0.9194

Epoch 5/100
----------
Train Loss: 0.4036 Acc: 0.8433
Val Loss: 0.3956 Acc: 0.9194

Epoch 6/100
----------
Train Loss: 0.3974 Acc: 0.8006
Val Loss: 0.3577 Acc: 0.8871

Epoch 7/100
----------
Train Loss: 0.3672 Acc: 0.8177
Val Loss: 0.4029 Acc: 0.8871

Epoch 8/100
----------
Train Loss: 0.3358 Acc: 0.8433
Val Loss: 0.3412 Acc: 0.8548

Epoch 9/100
----------
Train Loss: 0.3565 Acc: 0.8262
Val Loss: 0.4394 Acc: 0.7742

Epoch 10/100
----------
Train Loss: 0.3547 Acc: 0.8234
Val Loss: 0.2960 Acc: 0.9355

Epoch 11/100
----------
Train Loss: 0.3122 Acc: 0.8376
Val Loss: 0.2928 Acc: 0.9032

Epoch 12/100
----------
Train Loss: 0.2993 Acc: 0.8689
Val Loss: 0.3055 Acc: 0.9355

Epoch 13/100
----------
Train Loss: 0.3253 Acc: 0.8319
Val Loss: 0.3035 Acc: 0.8387

Epoch 14/100
----------
Train Loss: 0.3336 Acc: 0.8234
Val Loss: 0.3872 Acc: 0.8548

Epoch 15/100
----------
Train Loss: 0.3213 Acc: 0.8433
Val Loss: 0.3051 Acc: 0.9032

Epoch 16/100
----------
Train Loss: 0.2702 Acc: 0.8689
Val Loss: 0.3358 Acc: 0.8710

Epoch 17/100
----------
Train Loss: 0.3052 Acc: 0.8775
Val Loss: 0.3356 Acc: 0.8710

Epoch 18/100
----------
Train Loss: 0.2555 Acc: 0.8746
Val Loss: 0.3343 Acc: 0.8548

Epoch 19/100
----------
Train Loss: 0.2508 Acc: 0.8860
Val Loss: 0.3177 Acc: 0.8710

Epoch 20/100
----------
Train Loss: 0.2533 Acc: 0.8775
Val Loss: 0.3077 Acc: 0.8710

Epoch 21/100
----------
Train Loss: 0.2487 Acc: 0.8889
Val Loss: 0.3081 Acc: 0.8710

Epoch 22/100
----------
Train Loss: 0.2391 Acc: 0.8803
Val Loss: 0.2898 Acc: 0.8871

Epoch 23/100
----------
Train Loss: 0.2311 Acc: 0.8803
Val Loss: 0.3141 Acc: 0.9032

Epoch 24/100
----------
Train Loss: 0.2588 Acc: 0.8689
Val Loss: 0.2981 Acc: 0.8710

Epoch 25/100
----------
Train Loss: 0.2355 Acc: 0.8803
Val Loss: 0.2967 Acc: 0.8710

Early stopping triggered
Best val Acc: 0.935484

--- Test Results for squeezenet (binary) ---
Accuracy:  0.7961
Precision: 0.8750
Recall:    0.8116
F1 Score:  0.8421
Confusion Matrix:
[[26  8]
 [13 56]]


Training vgg for binary...
Epoch 1/100
----------
Train Loss: 0.6747 Acc: 0.6410
Val Loss: 0.6081 Acc: 0.4839

Epoch 2/100
----------
Train Loss: 0.5089 Acc: 0.7293
Val Loss: 0.3111 Acc: 0.8548

Epoch 3/100
----------
Train Loss: 0.3386 Acc: 0.8462
Val Loss: 0.3040 Acc: 0.8710

Epoch 4/100
----------
Train Loss: 0.3676 Acc: 0.8234
Val Loss: 0.3490 Acc: 0.8710

Epoch 5/100
----------
Train Loss: 0.3365 Acc: 0.8205
Val Loss: 0.2561 Acc: 0.8387

Epoch 6/100
----------
Train Loss: 0.3682 Acc: 0.8205
Val Loss: 0.4085 Acc: 0.7903

Epoch 7/100
----------
Train Loss: 0.3605 Acc: 0.8632
Val Loss: 0.2467 Acc: 0.8710

Epoch 8/100
----------
Train Loss: 0.2484 Acc: 0.8718
Val Loss: 0.2299 Acc: 0.9032

Epoch 9/100
----------
Train Loss: 0.2456 Acc: 0.8775
Val Loss: 0.3096 Acc: 0.8387

Epoch 10/100
----------
Train Loss: 0.2273 Acc: 0.8946
Val Loss: 0.2696 Acc: 0.8548

Epoch 11/100
----------
Train Loss: 0.1912 Acc: 0.9060
Val Loss: 0.3129 Acc: 0.8226

Epoch 12/100
----------
Train Loss: 0.1569 Acc: 0.9145
Val Loss: 0.5867 Acc: 0.8065

Epoch 13/100
----------
Train Loss: 0.2056 Acc: 0.9117
Val Loss: 0.4412 Acc: 0.7742

Epoch 14/100
----------
Train Loss: 0.2025 Acc: 0.9060
Val Loss: 0.8489 Acc: 0.6129

Epoch 15/100
----------
Train Loss: 0.2272 Acc: 0.8746
Val Loss: 0.3819 Acc: 0.7903

Epoch 16/100
----------
Train Loss: 0.1573 Acc: 0.9544
Val Loss: 0.3183 Acc: 0.8548

Epoch 17/100
----------
Train Loss: 0.1173 Acc: 0.9430
Val Loss: 0.3268 Acc: 0.8548

Epoch 18/100
----------
Train Loss: 0.1207 Acc: 0.9459
Val Loss: 0.2960 Acc: 0.8548

Epoch 19/100
----------
Train Loss: 0.0789 Acc: 0.9687
Val Loss: 0.3157 Acc: 0.8548

Epoch 20/100
----------
Train Loss: 0.1090 Acc: 0.9487
Val Loss: 0.3552 Acc: 0.8548

Epoch 21/100
----------
Train Loss: 0.0745 Acc: 0.9601
Val Loss: 0.3548 Acc: 0.8548

Epoch 22/100
----------
Train Loss: 0.0859 Acc: 0.9630
Val Loss: 0.3502 Acc: 0.8548

Epoch 23/100
----------
Train Loss: 0.0894 Acc: 0.9544
Val Loss: 0.3477 Acc: 0.8548

Early stopping triggered
Best val Acc: 0.903226

--- Test Results for vgg (binary) ---
Accuracy:  0.7670
Precision: 0.8261
Recall:    0.8261
F1 Score:  0.8261
Confusion Matrix:
[[22 12]
 [12 57]]

Done! Baseline training complete.
```
