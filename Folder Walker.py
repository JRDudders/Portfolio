"""This program was designed to prevent redundant files form being sent by our triage team to workflow. Although the majority of our work was tracked via a web portal,
some adhoc projects required us to manually search through our sent files folder tree to see if the files had been previously nominated for exploitation. This
program takes a spreadsheet filled with MD5 hashes and searches recursively to see if they're already in a given directory."""

import os, csv, xlsxwriter as xl, tkinter.filedialog, re

direc = tkinter.Tk()
direc.withdraw()
source = tkinter.filedialog.askopenfilename(parent=direc, initialdir="C:\\",title="Please select a CSV file")
directory = tkinter.filedialog.askdirectory(parent=direc, initialdir="C:\\",title="Please select a directory to search")
tracker = xl.Workbook('MD5 Checker.xlsx')
sheet = tracker.add_worksheet('Found Serials')

rownum = 0

with open(source, 'r', encoding='utf8') as f:
    contents = csv.reader(f)
serials = set()
for content in contents:
    if len(content) == 1:
        for hash in re.findall(r'[a-fA-F0-9]{32}', content[0]):
            serials.add(hash.upper())

    else:
        serials.add(content[0])
        serials.add(content[2])

print(f'Searching for {len(serials)} files...')

count = 0

for root, dirs, files in os.walk('\\\\?\\' + directory.replace('/', '\\')):
    for name in files:
        filepath = root + '\\' + name
        col = 0
        if name[:32].upper() in serials:
            count += 1
            print('Match on', filepath)
            sheet.write(rownum, col, name[:32])
            col += 1
            sheet.write(rownum, col, filepath)
            rownum += 1

print(f'Found {count}/{len(serials)} hashes')

