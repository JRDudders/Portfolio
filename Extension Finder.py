import os, datetime, xlsxwriter as xl, tkinter.filedialog

direc=tkinter.Tk()
direc.withdraw()
root_dir = tkinter.filedialog.askdirectory(parent=direc,initialdir="C:\\",title='Please select search folder')
tracker = xl.Workbook('Tool tracker 20201014.xlsx')
sheet = tracker.add_worksheet('Tools')
extensions = ['exe', 'bat', 'bin', 'cmd', 'zip', 'cpl', 'inf', 'ins', 'inx', 'isu', 'job', 'jse', 'lnk', 'msc', 'msi',
			  'msp', 'mst', 'paf', 'pif', 'ps1', 'reg', 'rgs', 'scr', 'sct', 'shb', 'shs', 'u3p', 'vb', 'vbe', 'vbs', 'vbscript',
			  'ws', 'wsf', 'wsh', 'as', 'ear', 'hta', 'iim', 'jar', 'js', 'jsx', 'mam', 'nexe', 'otm', 'potm', 'ppam', 'ppsm',
			  'pptm', 'pyc', 'pyo', 'udf', 'url', 'com', 'gadget', 'xbap', 'xlam', 'xlm', 'xlsm', 'xltm']

row = 0

for root, dirs, files in os.walk(root_dir):
	for name in files:
		try:
			filepath = os.path.join(root, name)
			name = os.path.basename(filepath)
			time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')

		except Exception as e:
			print(e)

		finally:
			col = 0
			if any(name.endswith(ext) for ext in extensions):
				print('Match on', filepath)
				sheet.write(row, col, name)
				col += 1
				sheet.write(row, col, filepath)
				col += 1
				sheet.write(row, col, time)
				row += 1

tracker.close()

