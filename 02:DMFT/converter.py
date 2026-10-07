from triqs_dft_tools.converters import Wannier90Converter

# Initialize converter using the seed name 'ybco'
converter = Wannier90Converter(seedname='ybco')

# Generate ybco.h5 from ybco_hr.dat, ybco.win, and ybco.inp
converter.convert_dft_input()