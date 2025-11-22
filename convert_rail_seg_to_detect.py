import os 
import shutil

src_root= "C:/Users/adalc/Downloads/proyecto vision artificial/RAIL.v3i.yolov8-obb"
dst_root= "C:/Users/adalc/Downloads/proyecto vision artificial"

splits = ["train", "valid", "test"]

for split in splits:
    img_src = os.path.join(src_root, split, "images")
    lbl_src = os.path.join(src_root, split, "labels")


    img_dst = os.path.join(dst_root, split, "images")
    lbl_dst = os.path.join(dst_root, split, "labels")

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    for fname in os.listdir(img_src):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy(os.path.join(img_src, fname),
                        os.path.join(img_dst, fname))
            
    for fname in os.listdir(lbl_src):
        if not fname.endswith(".txt"):
            continue

        in_path = os.path.join(lbl_src, fname)
        out_path = os.path.join(lbl_dst, fname)

        with open(in_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts)<5:
                continue

            new_line= " ".join(parts[:5]) + "\n"
            new_lines.append(new_line)

        with open(out_path, "w") as f:
            f.writelines(new_lines)
print("Conversion hecha: ",dst_root)




