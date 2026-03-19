import kagglehub

kitti_path = kagglehub.dataset_download("hannanguyen24/kitti-object-detection-2d")
sun_path = kagglehub.dataset_download("thanhbnhphan/sun-rgbd-2d")

print("KITTI:", kitti_path)
print("SUN:", sun_path)
