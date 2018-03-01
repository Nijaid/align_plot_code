from astropy.table import Table
from numpy import median

def ave(file):
    fi = Table.read(file, format='ascii')
    print('Strehl: ', fi['col2'].mean(), '\nRMS error (nm): ', fi['col3'].mean(), \
            '\nFWHM', fi['col4'].mean())

if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        ave(argv[1])
