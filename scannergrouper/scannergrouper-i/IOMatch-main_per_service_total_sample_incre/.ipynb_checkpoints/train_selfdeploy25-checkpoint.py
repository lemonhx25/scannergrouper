from datetime import datetime, timedelta
import subprocess
import os

result = subprocess.run(['echo', 'Hello, World!'], capture_output=True, text=True)
print(result.stdout)  


def generate_date_range(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        next_date = min(current_date + timedelta(days=7), end_date + timedelta(days=1))
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date = next_date
    return date_list

dataset_name='selfdeploy_24_25_1week'

batch_size_list=[8,16,32,64,128]
epoch_per_week_list=[1,2,3,4,5]

for batch_size in batch_size_list:
    for epoch_per_week in epoch_per_week_list:
        print('batch_size',batch_size,'epoch_per_week',epoch_per_week)
        for protocol in ['http','tls','dns']:
            start_date = '2024-12-10'    
            end_date = '2025-03-10'      
            
            date_ranges = generate_date_range(start_date, end_date)
            
            print(date_ranges)
            for current_date_index in range(len(date_ranges)):
                start_date_str=date_ranges[current_date_index]
                
                cmd1='rm data/'+dataset_name+'_'+protocol+'_openset/labeled_idx/*'
                

                date=start_date_str
                epoch=epoch_per_week*(current_date_index+1)
                save_name="iomatch_"+dataset_name+"_"+protocol+"_ep"+str(epoch_per_week)+"_bs"+str(batch_size)
                load_path="./saved_models/openset_cv/"+save_name+"/latest_model.pth"
                cmd2='python train.py --c config/openset_cv/iomatch/selfdeploy_24_25_'+protocol+'_incre.yaml' +' --epoch ' +str(epoch) +' --num_train_iter '+ str(epoch*100) +' --date ' +date +' --batch_size ' + str(batch_size) +' --save_name '+ save_name +' --load_path '+load_path
                print(cmd1,cmd2)
                r1=os.system(cmd1)
                print(r1)
                r2=os.system(cmd2)
                print(r2)
        
