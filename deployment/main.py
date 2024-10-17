#%%
import sys


src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)
#%%
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt

from deployment.colors_counter import Create, colors_counter
from deployment.symmetry_repair import symmetry_repairing
from deployment.sklearn_tree import format_features, make_features, tree1
from deployment.diff_solvers import check_chess, check_grid, check_grid_transforms, check_repeating, check_sub_grid_2x, check_sub_mask, check_subitem, check_tiles_shape, grid_filter, predict_chess, predict_grid_transforms, predict_repeating, predict_repeating_mask, predict_tiles_shape, predict_transforms_grid_2x
#%%

# ..................................................................................... 1
def ganswer_answer(ganswer):
    
    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j].tolist()
        
        if (ganswer_j not in answer):  
            answer.append(ganswer_j)   
            
    return answer

# ..................................................................................... 2
def ganswer_answer_1(ganswer):
    
    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j]
        
        if (ganswer_j not in answer):  
            answer.append(ganswer_j)   
            
    return answer

# ..................................................................................... 3
def prn_plus(prn, answer):
    
    for j in range(len(answer)):
        prn = prn + [answer[j]]  
        
        if (j == 0):
            prn = prn + [answer[j]]
            
    return prn

# ..................................................................................... 4
def prn_select_2(prn): 
    if (len(prn) > 2):
        
        value_list = []
        string_list = []
        for el in prn:
            value = 0
            for i in range(len(prn)):
                if el == prn[i]:
                    value +=1
            value_list.append(value)
            string_list.append(str(el))    
        
        prn_df  = pd.DataFrame({'prn': prn , 'value': value_list, 'string': string_list}) 
        prn_df1 = prn_df.drop_duplicates(subset=['string'])
        prn_df2 = prn_df1.sort_values(by='value', ascending=False)   
        
        prn = prn_df2['prn'].values.tolist()[:2]
        
    return prn

# ..................................................................................... 5
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]


def plot_pic(x):
    plt.imshow(np.array(x), cmap=cmap, norm=norm)
    plt.show()

#%%
def run_main_solvers(data_path, sample_path): 
    
    with open(sample_path,'r') as f:
        sub_solver = json.load(f) 
        
    # with open('/kaggle/working/sub_icecube.json' , 'r') as f:
    #     sub_icecube = json.load(f)

    # ...............................................................................
    with open(data_path,'r') as f:
        tasks_name = list(json.load(f).keys())
    
    with open(data_path,'r') as f:
        tasks_file = list(json.load(f).values())
        
    num_tasks_solved = 0
    # ...............................................................................
    for n in tqdm(range(len(tasks_name))):
        task = tasks_file[n]
        t = tasks_name[n]
            
        for i in range(len(task['test'])): 
            test_input = np.array(task['test'][i]['input'])
            prn = []
                
            # ............................................................................... 1 - Different Solvers       
            if check_repeating(task, True): 
                ganswer = predict_repeating(test_input)
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
            
            # ________________________________________________________
            if check_grid(task) and check_sub_grid_2x(task): 
                ganswer = predict_transforms_grid_2x(task, test_input)
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
            
            # ________________________________________________________
            if check_grid(task) and check_chess(task, False, True): 
                ganswer = predict_chess(grid_filter(test_input))
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
            
            # ________________________________________________________
            if check_tiles_shape(task, True): 
                ganswer = predict_tiles_shape(task, test_input)
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
            
            # ________________________________________________________
            if check_grid(task) and check_grid_transforms(task): 
                ganswer = predict_grid_transforms(task, test_input)
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
            
            # ________________________________________________________
            if check_sub_mask(task): 
                ganswer = predict_repeating_mask(test_input)
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
  
            if len(prn) > 0:
                print("Task: ", t, "Solver1 found")
            # # ............................................................................... 2 - Sklearn tree          
            if check_subitem(task):
                train_t = format_features(task)
                test_t = make_features(test_input) 
                ganswer = tree1(train_t, test_t, test_input)  
                
                if (ganswer!= []):
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer) 
                    print("Task: ", t, "Solver2 found")

            # ............................................................................... 3 - Symmetry Repairing       
            basic_task = Create(task, i) 
            ganswer = symmetry_repairing(basic_task)   
        
            if (ganswer != -1):
                answer = ganswer_answer_1(ganswer)
                prn = prn_plus(prn, answer) 
                print("Task: ", t, "Symmetry Repair found")

     
            # # ............................................................................... 4 - Colors Counter
            basic_task = Create(task, i) 
            answer = colors_counter(basic_task)   
            
            if (answer != -1):
                answer = [answer]
                prn = prn_plus(prn, answer) 
                print("Task: ", t, "Color Counter found")


            # ...............................................................................  Conclusion
            if (prn != []):  
                num_tasks_solved += 1
                prn = prn_select_2(prn)
                
                sub_solver[t][i]['attempt_1'] = prn[0]
                display(pd.DataFrame(data={'Answers for task':t, 'Items':i, 'Attempt':'1', 'Files':'test_challenges'},index=[n]))
                plot_pic(prn[0])
                
                if (len(prn)==2):
                    sub_solver[t][i]['attempt_2'] = prn[1]
                    display(pd.DataFrame(data={'Answers for task':t, 'Items':i, 'Attempt':'2', 'Files':'test_challenges'},index=[n]))
                    plot_pic(prn[1])

            # ............................................................................... 5 - ICECube
            # if (sub_solver[t][i]['attempt_1'] != [[0, 0], [0, 0]]):     
            #     if (sub_icecube[t][i]['attempt_1'] != [[0, 0], [0, 0]]):
                    
            #         if (sub_solver[t][i]['attempt_1'] != sub_icecube[t][i]['attempt_1']):
            #             sub_solver[t][i]['attempt_2'] =  sub_icecube[t][i]['attempt_1']
                    
            # if (sub_solver[t][i]['attempt_1'] == [[0, 0], [0, 0]]):
            #     if (sub_solver[t][i]['attempt_2'] == [[0, 0], [0, 0]]):
                    
            #         if (sub_icecube[t][i]['attempt_1'] != [[0, 0], [0, 0]]):
            #             sub_solver[t][i]['attempt_1'] = sub_icecube[t][i]['attempt_1']  
                        
            #         if (sub_icecube[t][i]['attempt_2'] != [[0, 0], [0, 0]]):
            #             sub_solver[t][i]['attempt_2'] = sub_icecube[t][i]['attempt_2']                
                    
    # ............................................................................... 
    # display(sub_solver)    
    print("Number of tasks solved: ", num_tasks_solved)
    return sub_solver

# ...............................................................................    

#%%
test_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_test_challenges.json'
sample_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/sample_submission.json'
sub_solver = run_main_solvers(test_path, sample_path)

with open('submission.json', 'w') as file:
    json.dump(sub_solver, file, indent=4)
# %%
