for lr in 1e-4 1e-2
do
    python run_Co-PSL_main.py --lr $lr --gpu 0
done