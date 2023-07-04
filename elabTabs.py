import os
from collections import defaultdict
import pandas as pd

num_fields =['EPOCHS','LAMB','ALPHA','FT','ID','WD', 'MOM', 'BS', 'THR', 'THRSTR', 'DIM', 'TIME']
optional_field = ['OPT', 'DIM', 'TIME']

def create_csv(path = "res/gurobi/"):
    name_list = os.listdir(path)
    
    table_cols = defaultdict(lambda: [])
    
    for log_name in name_list:
        print(log_name)
        f = open(path+log_name,'r')
        ls=f.readlines()
        for l in ls[1:]:
            s=log_name

            s=s.split('-')
            if s[-1][-4:] == '.csv':
                s[-1]= s[-1][:-4]
            optionals = ''

            for elem in s:

                # if 'FT' == elem[:2] and '_' not in elem:
                #     elem = 'FT_'+elem[2:]

                par, val = elem.split('_')
                if par in optional_field:
                    optionals += par + '=' + val + '\n'
                    continue

                if par in num_fields:
                    val = float(val)

                table_cols[par].append(val)
            table_cols['optionals'].append(optionals)
        


            # l=ls[1]
            l=l.split(',')
            table_cols['Acc'].append(float(l[0]))
            table_cols['Loss'].append(float(l[1]))
            table_cols['Time'].append(float(l[2]))
            table_cols['Nodes'].append(float(l[3]))
            table_cols['Pruned_arch'].append(l[4])

        f.close()


    table_cols = dict(table_cols)
    # breakpoint()
    table_cols_df = pd.DataFrame(table_cols)
    table_cols_df.to_csv(path  + "Tab.csv", sep = "\t", index = False)
    


key_words =['alpha','_reg','_ADPTno_','samp','eps','-lr','STRT']

def create_csv_temp(path = "res/gurobi/"):
    name_list = os.listdir(path)
    
    table_cols = defaultdict(lambda: [])
    df_tot = pd.DataFrame({'Acc':[],'Loss':[],'Elapsed':[],'Nodes':[],'Arch':[]})
    
    for log_name in name_list:
        df = pd.read_csv(path+log_name)
        df_tot=pd.concat([df_tot,df])
    
    return df_tot
    


