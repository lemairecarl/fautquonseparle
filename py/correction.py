with open('nouv.txt', 'r') as fnouv:
    lignes_nouv = list(fnouv.readlines())

with open('nouv_corr.txt', 'r') as cnouv:
    lignes_corr = list(cnouv.readlines())

with open('correction.txt', 'w') as f:
    for ln, lc in zip(lignes_nouv, lignes_corr):
        n = ln.split(',')[1].strip()
        c = lc.split(',')[1].strip()
        f.write('{},{}\n'.format(n, c))
        