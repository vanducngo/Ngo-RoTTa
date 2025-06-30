import csv

# Read the CSV file
with open('/Users/admin/Working/Data/MixData/vinbigdata_structured/validate.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Process each value
for i in range(len(data)):
    for j in range(len(data[i])):
        try:
            value = float(data[i][j])  # Convert to float to handle decimal comparison
            if value > 1:
                data[i][j] = '1'  # Change to 1
            else:
                data[i][j] = str(int(value))  # Convert to integer and then to string
        except:
            pass

# Write the processed data back to a new CSV file
with open('/Users/admin/Working/Data/MixData/vinbigdata_structured/validate_fix.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)