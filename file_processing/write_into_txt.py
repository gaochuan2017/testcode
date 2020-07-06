import os
file_folder_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/images"
write_dir="/home/gaochuan/object_detection/dataset/dota_8p"
txt_filename="test_train_8p.txt"
def write2txt(file_folder_dir,write_dir,txt_filename):
    for (d1,d2,d3) in os.walk(file_folder_dir):
        if(d1==file_folder_dir):
            with open(write_dir+os.sep+txt_filename,'a') as f:
                for file in d3:
                    line=file_folder_dir+os.sep+file
                    assert(os.path.exists(line))
                    f.write(line+'\n')

if __name__ == "__main__":
    write2txt(file_folder_dir=file_folder_dir,write_dir=write_dir,txt_filename=txt_filename)
