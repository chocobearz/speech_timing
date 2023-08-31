import os

directory = r"C:\\\Users\\\ptut0\\\Documents\\\speech_timing\\\basline\\\data\txt\\"  

# change to UTF-8
for file in os.listdir(directory):
  if ".txt" in file:
    with open(directory+file, 'r', encoding='cp1252') as f:
      script = f.read()
      script = script.replace("\n","")
      script = script.replace("’","'")
      script = script.replace(",.",'.')
      script = script.replace("  "," ")
      script = script.replace(" . ",". ")
      script = script.replace(" “",", ")
      script = script.replace("” ",", ")
      script = script.replace("”","")
      script = script.replace("-"," ")
      text_file = open(directory+"clean\\"+file, "w", encoding='utf-8')
      text_file.write(script)
      text_file.close()
      f.close()
