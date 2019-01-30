#python3 main.py train base_01 -batch_size 8 -epoch_num 100 -save ../weights/bese_01_01.pkl -log logs/base_01_01.txt -check_batch_num 20
#python3 main.py train fcn32s_01 -batch_size 6 -epoch_num 50 -save ../weights/fcn32s_01_01.pkl -log logs/fcn32s_01_01.txt -check_batch_num 20
python3 main.py train fcn32s_02 -batch_size 6 -epoch_num 100 -save ../weights/fcn32s_02_01.pkl -log logs/fcn32s_02_01.txt -check_batch_num 20
python3 main.py train fcn8s     -batch_size 6 -epoch_num 100 -save ../weights/fcn8s_01.pkl     -log logs/fcn8s.txt        -check_batch_num 20
