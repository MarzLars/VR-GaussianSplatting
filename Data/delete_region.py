import struct
import math

def read_ply_header(folder, filename):
    header = []
    with open(folder + filename, 'rb') as file:
        line = file.readline().decode().strip()
        while line != 'end_header':
            header.append(line)
            line = file.readline().decode().strip()
    return header

def update_vertex_count_in_header(header, new_count):
    # Update the vertex count in the header
    for i, line in enumerate(header):
        if line.startswith("element vertex"):
            original_count_length = len(line.split()[-1])
            new_count_str = str(new_count)
            num_spaces = original_count_length - len(new_count_str)
            header[i] = "element vertex " + " " * num_spaces + new_count_str
            break

    return header

def process_ply_data(folder, filename, new_header):
    new_filename = filename[1:]
    vertex_count = 0
    removed_count = 0
    with open(folder + filename, 'rb') as file, \
         open(folder + new_filename, 'wb') as new_file, \
         open(folder + "removed.ply", 'wb') as removed_file:
        # Write new header
        for line in new_header:
            new_file.write((line + '\n').encode())
        new_file.write(b'end_header\n')

        for line in new_header:
            removed_file.write((line + '\n').encode())
        removed_file.write(b'end_header\n')

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
            vertex_data = file.read(bytes_per_vertex)
            if not vertex_data:
                break
            x, y, z = struct.unpack(float_format * 3, vertex_data[:12])
            # Fox
            '''
            if (abs(x-2.02) <= 0.4 and abs(y+1.60) <= 0.4 and abs(z+0.186)<=0.4) or \
               (abs(x-2.9) <= 0.6 and abs(y-1.53) <= 0.6 and abs(z+0.72)<=0.6) or \
               (abs(x-2.23) <= 0.9 and abs(y+0.22) <= 0.9 and abs(z-1.50)<=0.9):
                continue
            '''
            # Bear
            '''
            if (abs(x+0.07) <= 2.3 and abs(y-0.176) <= 0.3 and abs(z-0.91)<=0.4) or \
               (abs(x-2.2) <= 1.35 and abs(y+0.25) <= 1.35 and abs(z+0.476)<= 1.35) or \
               (abs(x-1.298) <= 1.35 and abs(y-2.08) <= 1.35 and abs(z+1.98)<= 1.35) or \
               (abs(x-2.86) <= 0.6 and abs(y-1.46) <= 0.6 and abs(z+2.18)<= 0.6) or \
                (abs(x-4.1) <= 2.5 and abs(y+1.9) <= 2.5 and abs(z-2.05)<= 2.5):
                
            '''
            # Horse
            '''
            if (abs(x-1.19) <= 3.3 and abs(y+3.69) <= 3.3 and abs(z-1.16)<= 3.3) or \
                (-4.5<=x<=6.7 and -2.0<=y<=0.45 and -4.6 <= z <= 5.6):
            '''
            # 0
            # if (abs(x+0.254) <= 0.2 and abs(y-1.781) <= 0.2 and abs(z-0.678)<= 0.2):
            # if (abs(x-1.020) <= 1.0 and abs(y-1.766) <= 1.0 and abs(z-0.828)<= 1.0):

            # Teddy
            # if not ((abs(x-0.5) <= 2.0 and abs(y+0.22) <= 2.0 and abs(z-0.16)<= 2.0)):

            # Toys - Monkey
            # if not ((abs(x+3.7) <= 1.0 and abs(y-0.5) <= 1.0 and abs(z+1.24)<= 1.0)):
            # Toys - Bunny
            # if not ((abs(x+1.17) <= 1.5 and abs(y-0.32) <= 1.5 and abs(z+0.247)<= 1.5)):
            # if not ((abs(x+3.252) <= 1.0 and abs(y-0.017) <= 1.0 and abs(z+0.8911)<= 1.0)):

            # jls
            # th = 2.5
            # if (-th <= x <= th and -th <= z <= th and -1.5 <= y <= 1.23):

            # JY dance
            # th = 1.3
            # if not (-th <= x <= th and -th <= z <= th and -2.1 <= y <= 1.58):

            # Siyu dance
            # th = 1.9
            # if not (-th <= x <= th and -th <= z <= th and -1.9 <= y <= 1.95):

            # Revised basket
            # if ((-1.2 <= x <= 1.2 and 0.4 <= y <= 1.80 and -1.0 <= z <= 1.0)):
            # Revised table
            x0, z0 = -0.151622, 0.120026
            angle_degrees = -34
            angle_radians = math.radians(angle_degrees)
            x_prime = x0 + (x - x0) * math.cos(angle_radians) + (z - z0) * math.sin(angle_radians)
            z_prime = z0 - (x - x0) * math.sin(angle_radians) + (z - z0) * math.cos(angle_radians)
            if -2<=x_prime-x0<=2 and -2<=z_prime-z0<=2 and abs(y-0.353782) <= 0.6:
                vertex_count += 1
                new_file.write(vertex_data)
            else:
                removed_count += 1
                removed_file.write(vertex_data)

        new_header1 = new_header.copy()
        updated_header1 = update_vertex_count_in_header(new_header1, removed_count)
        removed_file.seek(0)
        for line in updated_header1:
            removed_file.write((line + '\n').encode())
        removed_file.write(b'end_header\n')

        new_header2 = new_header.copy()
        updated_header = update_vertex_count_in_header(new_header2, vertex_count)
        new_file.seek(0)
        for line in updated_header:
            new_file.write((line + '\n').encode())
        new_file.write(b'end_header\n')

    return new_filename

folder = '.\\revised_table\\'
filename = '_0_point_cloud.ply'
header = read_ply_header(folder, filename)
new_filename = process_ply_data(folder, filename, header)
print(f"Modified file saved as {new_filename}")
