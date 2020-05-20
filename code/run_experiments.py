import os
'''
use the link prediction tasks to evaluate the model
'''

def run_commands(commands, directory="../results/"):
    for python_cmd,file_out in commands:
        path_out = os.path.join(directory, file_out)
        cmd = "{} >> {}".format(python_cmd, path_out)
        report = "echo \"{}\" > {}".format(python_cmd, path_out)
        print(cmd)
        os.system(report)
        os.system(cmd)

directory = "../results/20_50/"

# python3.7 main.py -e 100 --attention_mode self --skip_mode add -d P_all_50 -f one_hot

commands = [
    ("python3.7 main.py -e 600 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_classify.out"),
    ("python3.7 main.py -e 600 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_classify_text.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 0 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_rel_0.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 1 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_rel_1.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 2 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_rel_2.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 3 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_rel_3.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 4 --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "single_rel_4.out"),
    ("python3.7 main.py -e 600 -t MultiTask --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "multi_task_arch1.out"),
    ("python3.7 main.py -e 600 -t MultitaskConcat --attention_mode self --skip_mode add -d P_all_50 -f one_hot", "multi_task_arch2.out")
]
run_commands(commands, directory)


directory = "../results/50/"

commands = [
    ("python3.7 main.py -e 600 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_classify.out"),
    ("python3.7 main.py -e 600 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_classify_text.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 0 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_rel_0.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 1 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_rel_1.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 2 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_rel_2.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 3 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_rel_3.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 4 --attention_mode self --skip_mode add -d P50 -f one_hot", "single_rel_4.out"),
    ("python3.7 main.py -e 600 -t MultiTask --attention_mode self --skip_mode add -d P50 -f one_hot", "multi_task_arch1.out"),
    ("python3.7 main.py -e 600 -t MultitaskConcat --attention_mode self --skip_mode add -d P50 -f one_hot", "multi_task_arch2.out")
]
run_commands(commands, directory)

directory = "../results/politicians/"

commands = [
    ("python3.7 main.py -e 600 --attention_mode self --skip_mode add -f one_hot", "single_classify.out"),
    ("python3.7 main.py -e 600 --attention_mode self --skip_mode add -f one_hot", "single_classify_text.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 0 --attention_mode self --skip_mode add -f one_hot", "single_rel_0.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 1 --attention_mode self --skip_mode add -f one_hot", "single_rel_1.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 2 --attention_mode self --skip_mode add -f one_hot", "single_rel_2.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 3 --attention_mode self --skip_mode add -f one_hot", "single_rel_3.out"),
    ("python3.7 main.py -e 600 -t SingleLink --single_relation 4 --attention_mode self --skip_mode add -f one_hot", "single_rel_4.out"),
    ("python3.7 main.py -e 600 -t MultiTask --attention_mode self --skip_mode add -f one_hot", "multi_task_arch1.out"),
    ("python3.7 main.py -e 600 -t MultitaskConcat --attention_mode self --skip_mode add -f one_hot", "multi_task_arch2.out")
]
run_commands(commands, directory)




