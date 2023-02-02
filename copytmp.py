import shutil

with open("filescomputers.txt", "r", encoding="UTF-8") as f:
    files = f.readlines()

# copy all files to /input_wiki_animal/
for file in files:
    file = file.strip()
    shutil.copyfile(file, "input_wiki_computers/" + file.split("\\")[-1])