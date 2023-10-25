import wget
import os

file = open('GSE223917_series_matrix.txt', mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()
urls = ""

# Parse through text file to get list of urls
for line in lines:
    if line[:28] == '!Sample_supplementary_file_2':
        urls = line
        print(line[:50])

# Create list by splitting on whitespace and replacing 'ftp://ftp...' with 'https://ftp...'
my_list = urls.split()[1:]
for i in range(len(my_list)):
    link = my_list[i]
    my_list[i] = 'https' + link[4:-1]

# Download files from my_list
print("downloading all links")
os.chdir("xyz") # path for dir we want to save files to
for link in my_list:
    wget.download(link)
print("complete!")