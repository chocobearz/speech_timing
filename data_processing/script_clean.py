import re
import os

directory = r"C:\\Users\\ptut0\\Documents\\speech_timing\\TEDLIUM_release-3\\speaker-adaptation\\test\\"  
    
for file in os.listdir(directory+"\stm\\"):
  filename = os.fsdecode(directory+"\stm\\"+file)
  with open(filename, "r") as f:
    long_script = []
    for line in f:
      no_tags = re.sub(r'^.*?>', '', line)
      no_tags = no_tags.replace(" <unk>",',')
      no_tags = no_tags.replace(" '","'")
      long_script.append(no_tags)
    joined_script = "".join(long_script)
    script = joined_script.replace("\n,",".")
    script = script.replace("\n","")
    clean_filename = file.replace(".stm", "")
    text_file = open(directory+"\\txt\\"+ clean_filename +".txt", "w+")
    text_file.write(script)
    text_file.close()
    f.close()
    