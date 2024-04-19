import csv
from collections import defaultdict
from itertools import zip_longest

class ResumeDataProcessor:
    def __init__(self, structured_content):
        self.content = structured_content
    
    def process_content(self):
        return self.flatten_content(self.content)

    def flatten_content(self, content):
        output = defaultdict(list)
        for section, items in content.items():
            if isinstance(items, dict):
                result = self.flatten_content(items)
                for header, lines in result.items():
                    output[header].extend(lines)
            elif isinstance(items, list):
                output[section].extend(items)
        return output
    
    def write_to_csv(self, data, filename):
        if not data:
            print("No data to write to CSV.")
            return

        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            headers = sorted(data.keys())
            writer.writerow(headers)
            rows = zip_longest(*[data[header] for header in headers], fillvalue='')
            for row in rows:
                writer.writerow(row)

        print(f"CSV file '{filename}' has been created with matched content.")
