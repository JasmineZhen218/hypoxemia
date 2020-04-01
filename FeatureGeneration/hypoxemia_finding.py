import csv
with open('events_train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    next(reader)
    for row in reader:
        row_info = row[0].split('[')
        row_info_2 = row_info[0].split(',')
        patient = row_info_2[1]
        record = row_info_2[2]
        positions = [int(s) for s in row_info[1][1:-1].split(', ')]
        pass