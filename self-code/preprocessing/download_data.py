"""
Reads 'files.txt' for files to download
Writes to 'download_cmds.txt' bash commands to download individual kaggle files
"""
import os
from pathlib import Path

fname = "files.txt"
base_dir = Path("parent series data directory")
path = base_dir / fname

lines = []
with open(path, 'r') as file:
    for line in file:
        if line != "\n":
            if ".dcm" in line:
                lines.append(line.rstrip())

# for i in range(0, 10):
#     print("-")
#     print(lines[i])
#     print("-")

page_token = lines[0]
os.system(f"cd {base_dir}")
with open("download_cmds.txt", 'w') as file:
    for i in range(len(lines)):
        line = lines[i]
        line = line.rsplit(',')[0].rsplit('/')
        folder_names = line[:-1]
        file_name = line[-1]
        folder_path = base_dir
        download_path = ""
        for na in folder_names:
            folder_path = folder_path / na
            download_path = download_path + '/' + na
            print(folder_path)
            os.system(f"md {folder_path}")
        download_path = (download_path + '/' + file_name).lstrip('/')
        folder_path = str(folder_path)
        folder_path = (folder_path.replace('\\', '/')).lstrip('/')        
        file.write(f"kaggle competitions download rsna-knee-abnormality-detection -f {download_path} -p {folder_path} \n")
os.system(f'move download_cmds.txt {base_dir}')        



