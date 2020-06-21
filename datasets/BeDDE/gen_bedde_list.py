import glob

data_root = '/media/xiaofeng/codes/LinuxFiles/defogging/for_github/Defogging_eval/BeDDE'
rec_file = './bedde_list.txt'
recFile = open(rec_file, 'w')

cityFolders = glob.glob('%s/*'%data_root)

for cityPath in cityFolders:
  print('Current city folder path: %s'%cityPath)
  imPaths = glob.glob('%s/fog/*.png'%(cityPath))
  print(imPaths)
  for imPath in imPaths:
    recFile.write('%s \n'%imPath)
recFile.close()