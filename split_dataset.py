import os
import shutil
import random

# Lokasi asal dataset
source_dir = r"C:\Users\M S I\Downloads\archive (7)\Garbage classification\Garbage classification"

# Lokasi tujuan dataset di proyek ML.NET kamu
base_target = r"C:\Users\M S I\OneDrive\Desktop\UTS ML 10222052 Maulia\SampahDetection\dataset"

# Folder train dan test
train_dir = os.path.join(base_target, "train")
test_dir = os.path.join(base_target, "test")

# Membuat folder train/test jika belum ada
for folder in [train_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Daftar subfolder kelas (misal cardboard, glass, dll)
classes = os.listdir(source_dir)

# Loop tiap kelas
for class_name in classes:
    src_class_dir = os.path.join(source_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)

    # Buat folder kelas di train & test
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Ambil semua file gambar
    images = os.listdir(src_class_dir)
    random.shuffle(images)

    # Hitung jumlah 80% untuk train dan 20% untuk test
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Salin file ke masing-masing folder
    for img in train_images:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(src_class_dir, img), os.path.join(test_class_dir, img))

print("✅ Dataset berhasil dibagi ke folder train dan test!")