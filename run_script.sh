#!/bin/bash
# Run the experiments
find . -name __pycache__ -type d -exec /bin/rm -rf {} +


python Main.py -b 256 -cl 115 -ch 125 -lcf 3 -hcf 5 -lcp 45 -hcp 50 -trans 6 -sce 10000 -l 0.0005 -eps 1 -bl 0 -rc 2 -load 0 -nn 10 -ed 0.0001 -center 1 -eeval 190 -ecompare 1 -lca 0 -nodes 10 -bss 7000000 -ooo 005_model -li 0 -hi 0.05 -fng TWO_RINGS_TRAIN_LOW_LOAD_GREEDY.txt -qb 300 -interfere 1 -fa 4 -pa 48  -ttt 1 -modelname 01_model_low -scene train -eee 0 0.01 0.02 0.03 0.04 0.05 0 0 0 0 0 0 0 0 0 0 0 0 0
