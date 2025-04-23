import os

from PcapProcessor import PcapProcessor


def process_all_pcaps(raw_pcap_dir, processed_csv_dir, pfx2as_csv_file_path, honeypot_ips):
    """
    Processes all PCAP files in the specified directory and saves the processed CSV files to the target directory.

    Parameters.
    raw_pcap_dir: str - Directory of raw PCAP files.
    processed_csv_dir: str - Directory where the processed CSV files are saved.
    pfx2as_csv_file_path: str - CSV path of the PFX2AS data file for AS number to IP prefix mapping.
    honeypot_ip: str - Honeypot IP address.
    """
    # Ensure that the processed CSV file directory exists, and if it doesn't then create the directory
    os.makedirs(processed_csv_dir, exist_ok=True)

    # Iterate through all files in the original PCAP file directory
    for filename in os.listdir(raw_pcap_dir):
        # Processing only PCAP files 
        if filename.endswith(".pcap"):
            pcap_file_path = os.path.join(raw_pcap_dir, filename)
            print(pcap_file_path)
            # Build the full path to the processed CSV file
            output_csv_file_path = os.path.join(
                processed_csv_dir, f"processed_{filename.split('.')[0]}.csv"
            )
            print(f"Processing {pcap_file_path}...")

            # Creating a PcapProcessor instance and processing the current PCAP file
            processor = PcapProcessor(
                pcap_file_path,
                pfx2as_csv_file_path,
                honeypot_ips,
                output_csv_file_path
            )
            processor.process_packets()
            print()

# Define the original PCAP file directory and the processed CSV file directory
raw_pcap_dir = '../dataset/SelfDeploy24/'
processed_csv_dir = 'dataset/processed_csv_SelfDeploy24/tmp'
pfx2as_csv_file_path = 'pfx2as/routeviews-rv2-20240720-1200.csv'
honeypot_ips = ['192.168.1.0', '192.168.2.0', '192.168.3.0', '192.168.4.0', '192.168.5.0']

# process all PCAP files
process_all_pcaps(raw_pcap_dir, processed_csv_dir, pfx2as_csv_file_path, honeypot_ips)
