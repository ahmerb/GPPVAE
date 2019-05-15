import csv
import sys


def gppvae(history, file, debug=False):
    history['vs_1'] = [h[0] for h in history['vs']]
    history['vs_2'] = [h[1] for h in history['vs']]
    del history['vs']
    history['vars_1'] = [h[0] for h in history['vars']]
    history['vars_2'] = [h[1] for h in history['vars']]
    del history['vars']
    fieldnames = history.keys()
    with open(file, 'w') as f:
        writer = csv.writer(sys.stderr) if debug else csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(zip(*list(history.values())))
