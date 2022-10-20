
import os
# 图片路径
pic_path="dish/10"

def rename():
    piclist=os.listdir(pic_path)
    total_num=len(piclist)
    i=1
    for pic in piclist:
        if pic.endswith(".jpg"):
            # 原图片命名格式
            old_path=os.path.join(os.path.abspath(pic_path),pic)
            # 新图片命名格式
            new_path=os.path.join(os.path.abspath(pic_path),'two'+format(str(i))+'.jpg')
            os.renames(old_path,new_path)
            print(u"把原图片命名格式："+old_path+u"转换为新图片命名格式："+new_path)
            i=i+1
            print("总共"+str(total_num)+"张图片被重命名为:" 'W'+format(str(i-1))+".jpg形式")
rename()



