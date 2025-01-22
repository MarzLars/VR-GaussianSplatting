import json
import numpy
with open('./materials/output/tet.json', 'r') as file:
    data = json.load(file)

elems = numpy.array(data[13][1][1]).reshape(-1, 4)
nodes = numpy.array(data[15][1][0][1][7][5])

with open('./materials/0_tetgen.txt', 'w') as file:
        file.write(f"{nodes.shape[0]} {elems.shape[0]}\n")
        for row in nodes: file.write(" ".join(map(str, row)) + "\n")
        for row in elems: file.write(" ".join(map(str, row)) + "\n")