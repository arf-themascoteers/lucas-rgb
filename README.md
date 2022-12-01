Linear
=====
bgr: 0.242

bgr + 1/b: 0.541
bgr + 1/g: 0.549
bgr + 1/r: 0.462

bgr + 1/b2: 0.52
bgr + 1/g2: 0.495
bgr + 1/r2: 0.410

bgr + 1/b + 1/g + 1/r: 0.644
bgr + 1/g2 + 1/b2 + 1/r2: 0.556

bgr + 1/g2 + 1/b2 + 1/r2 + 1/b + 1/g + 1/r: 0.654

NN
==
3X1 : 0.2
3X100X1: 0.54
3X100X1 - 1000 epochs: 0.64
+ inv-green: 0.553

kfold
========
>>> bgr only: 0.5430285010102671
> with inv green: 0.54427234100942


rf
==
bgr + soci : 0.681
bgr + soci : 0.675



