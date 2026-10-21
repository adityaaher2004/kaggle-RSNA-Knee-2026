## Tables
- train.csv (4407, 14), unique studies 4407; train_series.csv (24371, 5), unique series 24371, studies 4407
- train/series study sets equal: True
- test.csv (3, 1), test_series.csv (15, 5) (sample only)
- labels: ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
- studies with all 12 labels: 58; with some-but-not-all: 0; with none: 4349

## Series
- series/study: mean 5.53, median 5, min 3, max 14; dist {3: 1, 4: 675, 5: 2299, 6: 698, 7: 310, 8: 145, 9: 176, 10: 74, 11: 21, 12: 5, 13: 2, 14: 1}
- planes: {'Sagittal': 9864, 'Coronal': 8609, 'Axial': 5898}
- (Fluid_Sensitive, Fat_Suppression) counts: {(0, 0): 10361, (1, 1): 14010}
- plane x fluid: {('Axial', 0): 1179, ('Axial', 1): 4719, ('Coronal', 0): 3985, ('Coronal', 1): 4624, ('Sagittal', 0): 5197, ('Sagittal', 1): 4667}
- studies having all 3 planes: 4407/4407; series per plane per study mean {'Axial': 1.34, 'Coronal': 1.95, 'Sagittal': 2.24} max {'Axial': 7, 'Coronal': 5, 'Sagittal': 6}
- studies with >=1 fluid-sensitive series: 4407

## Gold labels (studies with provided labels)
- prevalence: {'ACL': 0.41, 'MCL': 0.16, 'Medial Meniscus': 0.45, 'Lateral Meniscus': 0.4, 'Medial OA': 0.26, 'Lateral OA': 0.19, 'PF OA': 0.36, 'Effusion': 0.6, 'Synovitis': 0.47, "Baker's": 0.21, 'Contusion': 0.33, 'Fracture': 0.31}
- positives per gold study: {1: 8, 2: 7, 3: 11, 4: 7, 5: 9, 6: 7, 7: 5, 8: 1, 9: 3}  (0 means an all-negative study)
- positive counts: {'ACL': 24, 'MCL': 9, 'Medial Meniscus': 26, 'Lateral Meniscus': 23, 'Medial OA': 15, 'Lateral OA': 11, 'PF OA': 21, 'Effusion': 35, 'Synovitis': 27, "Baker's": 12, 'Contusion': 19, 'Fracture': 18}
- label pairs with |phi|>0.4: Medial Meniscus~Medial OA 0.42, Lateral Meniscus~Medial OA 0.41, Lateral Meniscus~Lateral OA 0.42, Medial OA~Baker's 0.48, Lateral OA~Baker's 0.40, Effusion~Synovitis 0.40

## Reports
- chars: {'count': 4407.0, 'mean': 1098.0, 'std': 694.0, 'min': 52.0, '25%': 588.0, '50%': 977.0, '75%': 1460.0, 'max': 4743.0}
- exact-duplicate report texts: 177 rows (4276 unique)
- langdetect (seed 0) all: {'en': 1736, 'es': 682, 'tr': 546, 'hr': 406, 'el': 321, 'de': 262, 'bg': 220, 'nl': 153, 'fr': 81}
- langdetect gold: {'en': 28, 'es': 10, 'tr': 6, 'hr': 4, 'el': 3, 'bg': 3, 'nl': 2, 'de': 2}
- keyword hit fraction (EN/ES/DE stems only, crude): {ACL: 0.38, MCL: 0.37, Meniscus: 0.81, Effusion: 0.50, Baker: 0.31, Fracture: 0.18, Synovitis: 0.14}

## Local DICOM (headers only)
split  study series    plane  fluid  n      mfr  tesla  rows  cols  px_mm  fov_mm  thick  gap_mm  fname_order_is_anat  inst_vs_proj  bits  signed   slope      pid
train 817260 333772 Sagittal      1 22 Philips     1.5   512   512  0.330     169    3.4    4.63                False             1    12       0 2.55507 b9d20b75
train 817260 543265    Axial      1 18 Philips     1.5   560   560  0.288     161    4.0    7.72                False             1    12       0 2.41245 b9d20b75
train 817260 736174  Coronal      0 16 Philips     1.5   576   576  0.299     172    4.0    6.59                False             1    12       0 3.29524 b9d20b75
train 817260 476697  Coronal      1 16 Philips     1.5   560   560  0.307     172    4.0    6.59                False             1    12       0 2.91258 b9d20b75
train 817260 363546 Sagittal      0 22 Philips     1.5   640   640  0.264     169    3.0    4.63                False             1    12       0 2.64640 b9d20b75
train 399585 165237 Sagittal      0 34 Siemens     3.0   768   768  0.221     170    2.5    3.00                False             1    12       0 1.00000 ee084a79
train 399585 421711 Sagittal      1 34 Siemens     3.0   768   768  0.221     170    2.5    3.00                False             1    12       0 1.00000 ee084a79
train 399585 424266    Axial      1 38 Siemens     3.0   640   640  0.234     150    3.0    3.60                False            -1    12       0 1.00000 ee084a79
train 399585 147680  Coronal      0 30 Siemens     3.0   640   540  0.250     160    3.0    3.60                False             1    12       0 1.00000 ee084a79
train 399585 553280  Coronal      1 30 Siemens     3.0   640   540  0.250     160    3.0    3.60                False             1    12       0 1.00000 ee084a79
train 132231 921455  Coronal      1 38 Siemens     3.0   416   416  0.433     180    2.5    3.00                False             1    12       0 1.00000 f7648f12
train 132231 309042    Axial      1 48 Siemens     3.0   416   416  0.433     180    3.0    3.75                False            -1    12       0 1.00000 f7648f12
train 132231 984309  Coronal      0 38 Siemens     3.0   416   416  0.433     180    2.5    3.00                False             1    12       0 1.00000 f7648f12
train 132231 693623 Sagittal      0 38 Siemens     3.0   416   416  0.433     180    2.5    3.00                False            -1    12       0 1.00000 f7648f12
train 132231 479273 Sagittal      1 38 Siemens     3.0   416   416  0.433     180    2.5    3.00                False            -1    12       0 1.00000 f7648f12
 test 591402 048392 Sagittal      0 24 GE MEDIC    1.5   512   512  0.332     170    4.0    5.00                False            -1    16       1     NaN f04365d4
 test 591402 909030 Sagittal      1 24 GE MEDIC    1.5   512   512  0.332     170    4.0    5.00                False            -1    16       1     NaN f04365d4
 test 591402 539746    Axial      1 40 GE MEDIC    1.5   512   512  0.332     170    3.0    4.00                False            -1    16       1     NaN f04365d4
 test 591402 746544  Coronal      0 25 GE MEDIC    1.5   512   512  0.352     180    4.0    5.00                False            -1    16       1     NaN f04365d4
 test 591402 034312  Coronal      1 25 GE MEDIC    1.5   512   512  0.352     180    4.0    5.00                False            -1    16       1     NaN f04365d4

- series on disk 20; studies 4; patients 4; slices 598
- filename order == anatomical order in 0/20 series (0 expected: never sort by filename)
- InstanceNumber direction vs IPP-projection: +1 in 11, -1 in 9 series (sign is not constant)
- FOV mm range 150-180, pixel spacing 0.221-0.433, matrix 416-768, slices/series 16-48
- manufacturers {'Siemens ': 10, 'Philips ': 5, 'GE MEDIC': 5}, field strength {1.5: 10, 3.0: 10}, PixelRepresentation {0: 15, 1: 5}
