import os

path = '/home/myamya/project/image_files/images'
files = os.listdir(path)
print(files)
for i in range(len(files)):
    print(files[i]) 
    fname, ext = os.path.splitext(files[i])
    print(fname,ext)
    print(os.path.join(path,fname,ext))
    os.rename(os.path.join(path,fname+ext), os.path.join(path,str(i)+ext))

    
