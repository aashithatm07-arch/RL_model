@echo off
python train.py --steps 10 --warmup_epochs 1
python generate.py --model outputs/misinfocorrect
pause
