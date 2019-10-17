#!/usr/local/bin/python3

import sys
import os

def main():
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print('File path {} does not exist. Exiting.'.format(filepath))
        sys.exit()

    if not filepath.endswith('.obj'):
        print('File {} is not an obj file. Exiting.'.format(filepath))
        sys.exit()

    base = os.path.splitext(os.path.basename(filepath))[0]

    vertices = []
    faces = []
    with open(filepath) as obj:
        for line in obj:
            tokens = line.split(' ')
            if tokens[0] == 'v':
                vertices.append(tokens[1:4])
            elif tokens[0] == 'f':
                face = [s.split('//')[0] for s in tokens[1:]]
                faces.append(face)

    with open(base + '.poly', 'w') as poly:
        poly.write('# Part 1 - node list\n')
        poly.write('# Node count, 3 dim, no attribute, no boundary marker\n')
        poly.write(f'{len(vertices)} 3 0 0\n')
        for i in range(len(vertices)):
            v = vertices[i]
            poly.write(f'{i+1} {v[0]} {v[1]} {v[2]}')

        poly.write("\n")
        poly.write('# Part 2 - facet list\n')
        poly.write(f'{len(faces)} 0\n')
        poly.write('# facets\n')
        for f in faces:
            poly.write('1\n') # 1 polygon, no hole, no boundary marker
            poly.write(f'{len(f)} {" ".join(f)}\n')

        poly.write("\n")
        poly.write("# Part 3 - hole list\n")
        poly.write("0\t# no hole\n")

        poly.write("\n")
        poly.write("# Part 4 - region list\n")
        poly.write("0\t# no region\n")








if __name__ == '__main__':
    main()
