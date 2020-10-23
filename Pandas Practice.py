"""This program was written in Jupyter Notebook using Anaconda. It's goal is to reconcile a spreadsheet that our triage team was using to track
requests for service being manually maintained in the workcenter against a CSV export of the RFS database."""

import pandas as pd, tkinter.filedialog

#initial variables
direc = tkinter.Tk()
direc.withdraw()
source1 = tkinter.filedialog.askopenfile(parent=direc,initialdir="C:\\",title='Please choose the first Spreadsheet:')
source2 = tkinter.filedialog.askopenfile(parent=direc,initialdir="C:\\",title='Please choose the second Spreadsheet:')

#read in spreadsheets to pandas
tracker1 = pd.read_excel(source1)
tracker2 = pd.read_excel(source2)

#convert to dataframes
first = pd.DataFrame(tracker1)
second = pd.DataFrame(tracker2)

#comparison function
def dataframe_diff(df1, df2, which=None):
    compare_df = df1.merge(df2, indicator=True, how='outer', encoding='utf8')
    if which is None:
        new_df = compare_df[compare_df['_merge'] != 'both']

    else:
        new_df = compare_df[compare_df['_merge'] == which]
    return new_df

dataframe_diff(first, second)
