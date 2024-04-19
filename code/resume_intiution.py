import csv
from collections import defaultdict
from itertools import zip_longest

class ResumeDataProcessor:
    def __init__(self, structured_content):
        self.content = structured_content
    
    def process_content(self):
        """ Process and flatten the structured content for CSV conversion. """
        return self.flatten_content(self.content)

    def flatten_content(self, content):
        """ Recursively flatten structured JSON, filtering out empty sections. """
        output = defaultdict(list)
        for section, items in content.items():
            if isinstance(items, dict):
                # Recursive call to process nested dictionary
                result = self.flatten_content(items)
                for header, lines in result.items():
                    # Extend only if there's actual content, filtering empty lists or lists with only empty strings
                    if any(line.strip() for line in lines):
                        output[header].extend(lines)
            elif isinstance(items, list) and any(item.strip() for item in items):
                # Extend if the list has any non-empty, non-whitespace content
                output[section].extend(items)
        return output
    
    def write_to_csv(self, data, filename):
        """ Write the processed data to a CSV file. """
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