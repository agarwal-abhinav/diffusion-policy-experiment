import zarr
import imageio.v2 as imageio

# path to your zarr file
zarr_path = "data/planar_pushing_cotrain/sim_sim_tee_data_carbon_large.zarr"

# open zarr group
root = zarr.open(zarr_path, mode="r")

# navigate to data/overhead_camera
frames = root["data"]["overhead_camera"]  # usually shape (T, H, W, C)

# select first 100 frames
frames100 = frames[:100]  # numpy-like slicing

# save first frame as PNG
first_frame = frames100[0]
imageio.imwrite("overhead_first.png", first_frame)

# write all 100 frames to MP4
out_path = "overhead_first100.mp4"
fps = 5  # frames per second

with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
    for frame in frames100:
        writer.append_data(frame)

print("Saved first frame as overhead_first.png")
print(f"Saved first 100 frames as {out_path}")
