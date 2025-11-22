yaml_text = """path: /content/drive/MyDrive/rail_human_platform.v1i.yolov8

train: train/images
val: valid/images
test: test/images

nc: 1
names: [person]
"""
with open("C:/Users/adalc/Downloads/proyecto vision artificial/rail_human_platform.v1i.yolov8/data.yaml","w") as f:
    f.write(yaml_text)

print("person.yaml creado ✅")
