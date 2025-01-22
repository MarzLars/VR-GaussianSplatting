import struct

def read_ply_header(folder, filename):
    header = []
    with open(folder + filename, 'rb') as file:
        line = file.readline().decode().strip()
        while line != 'end_header':
            header.append(line)
            line = file.readline().decode().strip()
    return header

def remove_last_three_floats_from_header(header):
    # Find and remove the last three 'property float' lines
    float_properties = [line for line in header if line.startswith("property float")]
    for _ in range(3):
        if float_properties:
            header.remove(float_properties.pop())
    return header

def process_ply_data(folder, filename, new_header):
    new_filename = 'modified_' + filename
    with open(folder + filename, 'rb') as file, open(folder + new_filename, 'wb') as new_file:
        # Write new header
        for line in new_header:
            new_file.write((line + '\n').encode())
        new_file.write(b'end_header\n')

        # Skip old header
        while file.readline().strip() != b'end_header':
            pass

        # Calculate bytes to read and write per vertex
        float_format = 'f'
        bytes_per_float = struct.calcsize(float_format)
        total_floats = sum(1 for line in new_header if line.startswith("property float"))
        bytes_per_vertex = bytes_per_float * total_floats

        # Read, process, and write data
        while True:
            vertex_data = file.read(bytes_per_vertex + 3 * bytes_per_float)
            if not vertex_data:
                break
            new_file.write(vertex_data[:-3 * bytes_per_float])

    return new_filename

folder = '.\\box_moving\\'
filename = '1_point_cloud.ply'
header = read_ply_header(folder, filename)
new_header = remove_last_three_floats_from_header(header)
new_filename = process_ply_data(folder, filename, new_header)
print(f"Modified file saved as {new_filename}")
