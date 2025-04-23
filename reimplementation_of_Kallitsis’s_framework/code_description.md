# code description 

## data_label_merger
This script is used to load and merge multiple CSV files from a specified directory, annotate and clean the raw data with labeled data, and finally save the merged dataset. It includes data cleansing, processing of duplicate labels, data collection and merging, and saving the processed dataset as a CSV file.

## Optuna
This project mainly contains the tasks of deep learning and cluster analysis, using autoencoders for dimensionality reduction of the data and clustering algorithms for cluster analysis. The code uses the Pytorch framework to build the autoencoder model and evaluate the processed data, including the evaluation process for semi-supervised learning.

### Optuna dependency
- Python 3.x
- Pytorch
- Scikit-learn
- Numpy
- Pandas
- Matplotlib
- Optuna

## PcapProcessor
`PcapProcessor` is a class for processing and analyzing PCAP files. With this class, you can extract information about various network packets from PCAP files, correlate them with AS (Autonomous System) information, and ultimately generate detailed network activity reports.

### PcapProcessor Functions
- Read and parse PCAP files
- Extracts source IP, destination IP, port, protocol, IP ID, TTL and other information of packets.
- Correlates IP addresses with AS numbers
- Identifies honeypot activity
- Calculate network packet characteristics such as average arrival time, number of unique ports, number of unique addresses, IPID policy, etc.
- Generate a final data report and save it as a CSV file.

## process_all_pcaps
This script is used to batch process all PCAP files in a specified directory and save the processed results as a CSV file. The process includes reading PCAP files, parsing network packets, extracting key information, correlating it with AS (Autonomous System) information, and finally generating a detailed network activity report.

## convert_pfx2as_gz_to_csv
`Pfx2AsConverter` in convert_pfx2as_gz_to_csv.py is a Python class for processing BGP routing data. The class is primarily used to decompress Pfx2As files (usually in `.gz` format) and convert them into an easy-to-analyze CSV file format. By converting data from compressed format to CSV format, users can analyze and process the data more easily.