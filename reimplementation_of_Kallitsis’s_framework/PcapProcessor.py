import time
import bisect
import ipaddress
import pandas as pd
from scapy.all import rdpcap
from scapy.layers.inet import TCP, IP, UDP, ICMP
import pickle
from tqdm import tqdm


class PcapProcessor:
    """
    process all PCAP files for analyzing PCAP data and generating reports.

    Parameters.
    pcap_file_path: str, PCAP file path.
    pfx2as_csv_file_path: str, CSV path of PFX2AS data file for AS number to IP prefix mapping.
    honeypot_ip: str, honeypot IP address.
    output_csv_file_path: str, CSV file path for output reports.
    """

    def __init__(self, pcap_file_path, pfx2as_csv_file_path, honeypot_ips, output_csv_file_path,
                 cache_file_path='prefix_cache.pkl'):
        """
        Initialize the method, load the necessary data files, and prepare the data structures to store the analysis results.
        """
        self.pcap_file_path = pcap_file_path
        self.pfx2as_csv_file_path = pfx2as_csv_file_path
        self.honeypot_ips = honeypot_ips  
        self.output_csv_file_path = output_csv_file_path
        self.cache_file_path = cache_file_path
        self.pfx2as_df = self.load_pfx2as_data(self.pfx2as_csv_file_path)
        self.data = {}
        self.sessions = {}
        self.prefix_counter = {}
        self.start_time = time.time()
        self.prefix_cache = self.load_cache()

    def load_pfx2as_data(self, csv_file_path):
        """
        Loads data prefixed to an AS number from a CSV file and processes it.

        Parameters.
        csv_file_path: str - Path to the CSV file containing the prefix and AS number data.

        Returns.
        pandas.DataFrame - Processed data frame with prefix, network, start address, end address and AS number.
        """
        pfx2as_df = pd.read_csv(csv_file_path)

        # Merge prefixes and prefix lengths into network objects
        pfx2as_df['network'] = pfx2as_df.apply(
            lambda row: ipaddress.ip_network(f"{row['prefix']}/{row['prefix_length']}"), axis=1)

        # Extract the starting address of the network (network address) as an integer
        pfx2as_df['start'] = pfx2as_df['network'].apply(lambda net: int(net.network_address))

        # Extract the end address of the network (broadcast address) as an integer
        pfx2as_df['end'] = pfx2as_df['network'].apply(lambda net: int(net.broadcast_address))

        # Sort the data frame according to start address and prefix length and reset the indexes
        pfx2as_df = pfx2as_df.sort_values(by=['start', 'prefix_length'], ascending=[True, False]).reset_index(drop=True)

        # Returns the processed data frame
        return pfx2as_df

    def load_cache(self):
        """
        Load Cached Data.

        Attempts to read and load cached data from the specified cache file path. If the file does not exist, an empty dictionary is returned.

        Returns.
            Returns the loaded cache data, or an empty dictionary if the file does not exist.
        """
        try:
            with open(self.cache_file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_cache(self):
        """
        Saves the prefix cache to a file.

        This method uses the pickle module to serialize the prefix_cache object and write it to the specified cache file.
        The purpose of this is to persist the cache data so that the cache can be loaded quickly on subsequent runs of the program, without the need to recalculate it.

        Arguments: self: the caller of the method.
        - self: the caller of the method, containing the prefix_cache and cache_file_path attributes.
        
        """
        with open(self.cache_file_path, 'wb') as f:
            pickle.dump(self.prefix_cache, f)

    def match_prefix(self, ip):
        """
        Finds and returns the network prefix and prefix length that matches the given IP address.

        It first looks up the prefix from the cache and returns the result directly if it exists. If it does not exist in the cache
        then determine the matching network prefix by looking up the network range in the data table.

        Parameters: ip (str)
        ip (str): The IP address to look up the matching prefix.

        Returns: ip (str)
        tuple: a tuple of matching network prefixes and prefix lengths, or (None, None) if no match is found.
        """
        # Check if there are already matches in the cache
        if ip in self.prefix_cache:
            return self.prefix_cache[ip]

        # Converts IP addresses to integer form for comparison purposes
        ip_addr = int(ipaddress.ip_address(ip))
        # Get a list of all network start addresses for subsequent binary lookups
        starts = self.pfx2as_df['start'].tolist()
        # Use a binary lookup to determine the index of the network range that a given IP address should belong to
        idx = bisect.bisect_right(starts, ip_addr) - 1

        # Starting at the found index, search forward until a matching network is found or the search ends
        while idx >= 0:
            # Get information about the currently checked network
            network = self.pfx2as_df.loc[idx, 'network']
            # Check if the current IP address falls within the current network range
            if ipaddress.ip_address(ip) in network:
                # If a match is found, update the cache and return the result
                self.prefix_cache[ip] = (str(network.network_address), network.prefixlen)
                self.save_cache()
                return self.prefix_cache[ip]
            # If the current network does not match, continue checking the previous network
            idx -= 1

        # If no match is found, cache empty results to improve the speed of subsequent queries
        self.prefix_cache[ip] = (None, None)
        self.save_cache()
        return None, None

    def process_packets(self):
        """
        Processes the packets in the pcap file.

        The method first reads all packets from the pcap file using the rdpcap function. It then iterates through each packet
        and calls the process_packet method to process each packet. For every 1000 packets processed during processing, the
        method prints the current processing progress and the time since processing began. After all packets have been processed, the method will call the
        calculate_features to calculate certain features and create a DataFrame using create_dataframe.
        Finally, it saves this DataFrame as a CSV file.

        The method does not accept any parameters and does not return any values.
        
        """
        print(f"The number of IP addresses in the cache is: {len(self.prefix_cache)}")
        # Read all packets from pcap file
        packets = rdpcap(self.pcap_file_path)

        # Using the tqdm library to display progress bars
        for idx, packet in tqdm(enumerate(packets), total=len(packets), desc="Processing packets"):
            self.process_packet(packet, idx + 1)

        # Calculate certain characteristics of the processed packets
        self.calculate_features()

        # Creating a DataFrame and saving it as a CSV file
        df = self.create_dataframe()
        df.to_csv(self.output_csv_file_path, index=False)
        print(f"CSV file saved to {self.output_csv_file_path}")

    def process_packet(self, packet, packet_no):
        """
        Processes the packet and records the relevant information.

        If the packet contains an IP layer, various attributes of the IP layer are extracted, including source IP, destination IP, protocol type, IP ID, and TTL.
        If the source IP is a honeypot IP, it is not processed further.
        Extract port and protocol type information.
        If the source IP has not been initialized, initialize the relevant data.
        If the packet contains a TCP option, record this information.
        Record the packet number using the source port, destination IP, and destination port as session keys.
        If the packet is the first packet of a specific protocol type and has not yet been logged, update the scanner data.

        Parameters.
        packet -- The parsed packet object.
        packet_no -- The packet number.
        """
        # Check if the packet contains the IP layer
        if packet.haslayer(IP):
            # Extract IP layer information
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            proto = ip_layer.proto
            ipid = ip_layer.id
            ttl = ip_layer.ttl

            # If the source IP is a honeypot IP, it is not processed
            if src_ip in self.honeypot_ips:
                return

            # Extract port and protocol type information
            src_port, dst_port, proto_type, tcp_options = self.extract_ports_and_protocol(packet)

            # If proto_type is not the specified protocol type, the data is not updated
            if proto_type is None or (
                    proto_type not in ['TCP-SYN', 'UDP', 'ICMP Echo Request', 'ICMP Dest. Unreachable',
                                       'ICMP Echo Reply','ICMP Redirect Message', 'ICMP Time Exceeded'] and not proto_type.startswith('ICMP')):
                return

            # If the source IP data is not initialized, initialize the
            if src_ip not in self.data:
                self.initialize_scanner_data(src_ip)

            # If the packet contains the TCP option, record the data
            if tcp_options:
                self.data[src_ip]['TCP Options'] = True

            # Generate session keys
            session_key = (src_port, dst_ip, dst_port)

            # If the session key is not recorded, initialize the session data
            if session_key not in self.sessions[src_ip]:
                self.sessions[src_ip][session_key] = []

            # Record the packet number into the session
            self.sessions[src_ip][session_key].append(packet_no)

            # If it is the first packet of a specific protocol type and has not yet been logged, update the scanner data
            if proto_type in ['TCP-SYN', 'UDP', 'ICMP Echo Request', 'ICMP Dest. Unreachable', 'ICMP Echo Reply',
                              'ICMP Redirect Message', 'ICMP Time Exceeded'] and packet_no not in self.data[src_ip][
                'Session First Packet Nos']:
                self.update_scanner_data(src_ip, packet, packet_no, dst_port, dst_ip, ipid, ttl, proto_type)

    def extract_ports_and_protocol(self, packet):
        """
        Extracts source port, destination port and protocol type information from a packet.

        For packets containing TCP, UDP or ICMP layers, this function will extract the corresponding port information and protocol type.
        For other types of packets, the protocol type will be labeled as 'Other'.

        Parameters.
        packet -- A Scapy packet to analyze port and protocol type information.

        Returns.
        src_port -- The source port number of the packet.
        dst_port -- The destination port number of the packet.
        proto_type -- The protocol type of the packet.
        tcp_options -- Boolean value of whether the options field is included in the TCP packet.
        
        """
        # Initialize port and protocol type variables
        src_port = None
        dst_port = None
        proto_type = None
        tcp_options = False

        # Check if the packet contains the TCP layer
        if packet.haslayer(TCP):
            # Extracting TCP layer information
            tcp_layer = packet[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            # Checking for the presence of the TCP option
            tcp_options = bool(tcp_layer.options)
            # Initially set the protocol type to TCP-SYN
            proto_type = 'TCP-SYN'
            # If the TCP flag bit does not contain the SYN flag, the protocol type is reset to None
            if not tcp_layer.flags & 2:
                proto_type = None
        # Check if the UDP layer is included in the packet
        elif packet.haslayer(UDP):
            # Extracting UDP layer information
            udp_layer = packet[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
            # Set protocol type to UDP
            proto_type = 'UDP'
        # Check if the packet contains the ICMP layer
        elif packet.haslayer(ICMP):
            # Extracts ICMP layer information and converts it to readable string form
            icmp_layer = packet[ICMP]
            proto_type = self.icmp_type_to_string(icmp_layer.type)
        # For other types of packets
        else:
            # Set the protocol type to Other
            proto_type = 'Other'

        # Returns extracted port and protocol information
        return src_port, dst_port, proto_type, tcp_options

    def icmp_type_to_string(self, icmp_type):
        if icmp_type == 8:
            return 'ICMP Echo Request'
        elif icmp_type == 3:
            return 'ICMP Dest. Unreachable'
        elif icmp_type == 0:
            return 'ICMP Echo Reply'
        elif icmp_type == 5:
            return 'ICMP Redirect Message'
        elif icmp_type == 11:
            return 'ICMP Time Exceeded'
        else:
            return f'ICMP Type {icmp_type}'

    def initialize_scanner_data(self, src_ip):
        self.data[src_ip] = {
            'Total Packets': 0,  
            'Total Bytes': 0,  
            'Inter-arrival Times': [],  
            'Distinct Ports': set(),  
            'Distinct Addresses': set(),  
            'IPID Values': [],  
            'TCP Options': False,  
            'Timestamps': [],  
            'Destination Strategy': set(),  
            'Set of Ports Scanned': set(),  
            'Protocol Types': set(),  
            'Prefix Density': None,  
            'Packet Nos': [], 
            'Session First Packet Nos': [],  
            'All TTL Values': [],  
            'Prefix': None,  
            'Prefix Length': None  
        }
        
        self.sessions[src_ip] = {}

    def update_scanner_data(self, src_ip, packet, packet_no, dst_port, dst_ip, ipid, ttl, proto_type):
        self.data[src_ip]['Packet Nos'].append(packet_no)
        self.data[src_ip]['Session First Packet Nos'].append(packet_no)

        self.data[src_ip]['Total Packets'] += 1
        self.data[src_ip]['Total Bytes'] += len(packet)

        
        if dst_port:
            self.data[src_ip]['Distinct Ports'].add(dst_port)
            self.data[src_ip]['Set of Ports Scanned'].add(dst_port)

        
        self.data[src_ip]['Distinct Addresses'].add(dst_ip)

        
        self.data[src_ip]['IPID Values'].append(ipid)

        
        self.data[src_ip]['All TTL Values'].append(ttl)

        # 
        self.data[src_ip]['Timestamps'].append(packet.time)

        # 
        self.data[src_ip]['Destination Strategy'].add(dst_ip)

        # 
        self.data[src_ip]['Protocol Types'].add(proto_type)

        # 
        prefix, prefix_length = self.match_prefix(src_ip)

        # 
        self.data[src_ip]['Prefix'] = prefix
        self.data[src_ip]['Prefix Length'] = prefix_length
        self.data[src_ip]['Network Prefix'] = f"{prefix}/{prefix_length}" if prefix and prefix_length else None

        # 
        if prefix:
            if prefix not in self.prefix_counter:
                self.prefix_counter[prefix] = set()
            self.prefix_counter[prefix].add(src_ip)

    def calculate_features(self):
        """

        """

        for src_ip, features in self.data.items():
            # 
            timestamps = features['Timestamps']
            # 
            if len(timestamps) > 1:
                inter_arrival_times = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                # 
                avg_inter_arrival_time = sum(inter_arrival_times) / len(inter_arrival_times)
            else:
                avg_inter_arrival_time = 0
            # 
            features['Average Inter-arrival Time'] = avg_inter_arrival_time
            # 
            features['Distinct Ports'] = len(features['Distinct Ports'])
            # 
            features['Distinct Addresses'] = len(features['Distinct Addresses'])
            # 
            features['IPID Strategy'] = self.detect_ipid_strategy(features['IPID Values'])
            # 
            features['Destination Strategy'] = self.detect_destination_strategy(features['Destination Strategy'])
            # 
            features['All TTL Values'] = features['All TTL Values']
            # 
            features['Device Type'] = self.identify_device_or_scanner_type(features['All TTL Values'])
            # 
            if features['Prefix'] and features['Prefix Length']:
                # 
                prefix_ip_count = 2 ** (32 - features['Prefix Length'])
                # 
                scanner_count_in_prefix = len(self.prefix_counter[features['Prefix']])
                # 
                features['Prefix Density'] = scanner_count_in_prefix / prefix_ip_count
            else:
                features['Prefix Density'] = 0

    def detect_ipid_strategy(self, ipid_list):
        """
        """
        # 
        if len(ipid_list) == 1 or all(ipid == ipid_list[0] for ipid in ipid_list):
            return 'Fixed Value'

        # 
        if len(ipid_list) > 2:
            # 
            increments = [ipid_list[i] - ipid_list[i - 1] for i in range(1, len(ipid_list))]
            # 
            if all(increment == increments[0] for increment in increments):
                return 'Fixed Increment'

        # 
        return 'Random Value'

    def detect_destination_strategy(self, destination_list):
        """
        
        """
        # 
        destination_list = list(destination_list)

        # 
        if all(dst_ip == destination_list[0] for dst_ip in destination_list):
            return 'Fixed Value'

        # 
        increments = [
            int(destination_list[i].split('.')[-1]) - int(destination_list[i - 1].split('.')[-1])
            for i in range(1, len(destination_list))
        ]

        # 
        if all(increment == increments[0] for increment in increments):
            return 'Fixed Increment'

        # 
        return 'Random Value'

    def identify_device_or_scanner_type(self, ttl_values):
        """

        """

        # 
        if any(40 <= ttl <= 60 for ttl in ttl_values):
            return 'Linux/Unix Device'

        # 
        elif any(100 <= ttl <= 120 for ttl in ttl_values):
            return 'Windows Device'

        # 
        else:
            return 'Unknown Device'

    def create_dataframe(self):
        """

        """
        # 
        rows = []

        # 
        for src_ip, features in self.data.items():
            # 
            rows.append([
                src_ip,
                features['Total Packets'],  # 
                features['Total Bytes'],  # 
                features['Average Inter-arrival Time'],  # 
                features['Distinct Ports'],  # 
                features['Distinct Addresses'],  # 
                features['Network Prefix'],  # 
                features['Prefix Density'],  # 
                features['Destination Strategy'],  # 
                features['IPID Strategy'],  # 
                features['TCP Options'],  # 
                features['Set of Ports Scanned'],  # 
                features['Protocol Types'],  # 
                features['All TTL Values'],  # 
                features['Device Type'],  # 
                features['Session First Packet Nos']  # 

            ])

        # 
        df = pd.DataFrame(rows, columns=[
            'Scanner IP', 'Total Packets', 'Total Bytes', 'Average Inter-arrival Time',
            'Number of Distinct Destination Ports', 'Number of Distinct Destination Addresses', 'Network Prefix',
            'Prefix Density', 'Destination Strategy', 'IPID Strategy', 'TCP Options',
            'Set of Ports Scanned', 'Protocol Types', 'Set of TTL Values', 'Device Type', 'Session First Packet Nos'
        ])

        # 
        return df

#processor = PcapProcessor(
#    'dataset/raw_pcap/pcap_all/2024_0217.pcap',
#    'pfx2as/routeviews-rv2-20240720-1200.csv',
#    ['192.168.1.0', '192.168.2.0', '192.168.3.0', '192.168.4.0', '192.168.5.0'],
#    'processed_2024_0217.csv'
#)
#processor.process_packets()
