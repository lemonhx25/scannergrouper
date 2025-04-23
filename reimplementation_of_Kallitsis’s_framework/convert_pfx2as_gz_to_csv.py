import gzip
import shutil
import pandas as pd

class Pfx2AsConverter:
    """
    
    Pfx2As file conversion class for decompressing pfx2as gz files and converting them to CSV files.

    Parameters.
    gz_file_path (str): path of the input gz file.
    """

    def __init__(self, gz_file_path):
        self.gz_file_path = gz_file_path
        self.txt_file_path = gz_file_path.replace('.gz', '')
        self.csv_file_path = gz_file_path.replace('.gz', '.csv')

    def convert(self):
        # Unzip the gz file
        with gzip.open(self.gz_file_path, 'rb') as f_in:
            with open(self.txt_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        with open(self.txt_file_path, 'r') as f:
            lines = f.readlines()
            print(f"Number of lines in {self.txt_file_path}: {len(lines)}")

        # Read the unzipped data file and make sure the delimiters are correct
        pfx2as_df = pd.read_csv(self.txt_file_path, sep='\t', names=['prefix', 'prefix_length', 'asn'])

        
        print(f"Number of rows in DataFrame: {len(pfx2as_df)}")

        # Save data to csv file
        pfx2as_df.to_csv(self.csv_file_path, index=False)

        print(f"CSV file saved to {self.csv_file_path}")


gz_file_path = 'pfx2as/routeviews-rv2-20240720-1200.pfx2as.gz'  


converter = Pfx2AsConverter(gz_file_path)
converter.convert()
