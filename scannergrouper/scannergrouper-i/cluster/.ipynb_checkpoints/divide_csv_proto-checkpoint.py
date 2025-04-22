import pandas as pd

#Specify dataset name
dataset_name='selfdeploy24_incre'

# Read raw CSV files
input_file = 'result_by_service/proj_'+dataset_name+'_withindex.csv'
input_data = pd.read_csv(input_file)

# Filtering out http, tls, dns data respectively
protocols = ['http', 'tls', 'dns']
for proto in protocols:
    filtered_data = input_data[input_data['proto'] == proto]
    print(filtered_data.head())
    # Save the filtered data to a new CSV file
    output_file = f'result_by_service/proj_{proto}_{dataset_name}_withindex.csv'
    filtered_data.to_csv(output_file, index=False)
    print(f"Data for protocol '{proto}' has been written to {output_file}")

    input_file_per_service = 'result_by_service/proj_'+proto+'_'+dataset_name+'_withindex.csv'
    data = pd.read_csv(input_file_per_service)
    
    condition = ((data['y_true'] == 26) | (data['y_true'] == 25)| (data['y_true'] == 27)) & (data['ip_pred'] == 25)
    data_y_true_26 = data[condition]
    
    # Save the segmented unknown data to a new CSV file
    output_file_26 = 'result_by_service/proj_still_unknown_'+proto+'_'+dataset_name+'_withindex.csv'
    
    data_y_true_26.to_csv(output_file_26, index=False)
    
    print(f"Data with y_true equal to 26 has been written to {output_file_26}")

print("All files have been created.")
