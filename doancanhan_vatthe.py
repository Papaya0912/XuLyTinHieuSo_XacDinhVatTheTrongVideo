# Nguồn tham khảo: CT144E@NguyenThanhTung

# Thư viện cần thiết
import os                                                # quản lý thư mục và tệp tin
import numpy as np                                       # xử lý mảng ảnh dạng ma trận số
import matplotlib.pyplot as plt                          # vẽ biểu đồ và chọn ROI
from matplotlib.widgets import RectangleSelector         # chọn vùng hình chữ nhật
from PIL import Image, ImageDraw, ImageFont              # xử lý ảnh, lưu ảnh, thêm chữ
from moviepy import VideoFileClip, ImageSequenceClip     # đọc video, xử lý video và tạo video từ chuỗi ảnh
from scipy.signal import correlate2d                     # tính toán tương quan 2D  

# Cấu hình đường dẫn 
workspace = os.getcwd()                                          # thư mục làm việc hiện tại
video_name = "trunggatrang5s.mp4"                                # tên file video đầu vào    
video_path = os.path.join(workspace, "VideoGoc", video_name)     # đường dẫn video đầu vào
output_base = os.path.join(workspace)                            # thư mục lưu kết quả
gray_folder = os.path.join(output_base, "AnhXam")                # thư mục lưu ảnh xám
color_folder = os.path.join(output_base, "AnhMau")               # thư mục lưu ảnh màu
output_folder = os.path.join(output_base, "VideoDaXuly")         # thư mục lưu video đầu ra

os.makedirs(gray_folder, exist_ok=True)     # tạo thư mục lưu ảnh xám
os.makedirs(color_folder, exist_ok=True)    # tạo thư mục lưu ảnh màu
os.makedirs(output_folder, exist_ok=True)   # tạo thư mục lưu video đầu ra

roi_array = None               # mảng lưu vùng chọn ROI
positions = []                 # danh sách chứa toạ độ x, y của vật thể theo khung hình

# Hàm chọn vùng ROI
def roi_selection(eclick, erelease):                                                
    global roi_array                                                                # biến toàn cục roi_array
    x1, y1 = int(eclick.xdata), int(eclick.ydata)                                   # toạ độ điểm click chuột     
    x2, y2 = int(erelease.xdata), int(erelease.ydata)                               # toạ độ điểm thả chuột
    roi_array = first_frame_gray[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]  # cắt vùng ROI từ khung hình đầu tiên
    plt.close()                                                                     # đóng cửa sổ chọn vùng   

# Bước 1: Trích xuất khung hình và chọn ROI
print("- Đang tải video...")                            
clip = VideoFileClip(video_path)                            # mở video
fps = clip.fps                                              # lấy số khung hình trên giây
timestamps = np.arange(0, clip.duration - 1/fps, 1/fps)     # mốc thời gian cho mỗi khung hình
print(f" - Thời lượng video: {clip.duration:.2f} giây | {len(timestamps)} khung hình | {fps:.2f} fps")  # in ra thông tin video

first_frame = clip.get_frame(0)                                             # lấy khung hình đầu tiên
first_frame_gray = np.array(Image.fromarray(first_frame).convert('L'))      # chuyển sang ảnh xám

print("- Chọn vùng vật thể (ROI) trên khung hình đầu tiên")         
fig, ax = plt.subplots(figsize=(10, 7))                         # tạo cửa sổ hiển thị
ax.set_title("Chọn vùng vật thể (Click và kéo)")                # tiêu đề cửa sổ
ax.imshow(first_frame_gray, cmap='gray')                        # hiển thị ảnh xám 
rs = RectangleSelector(ax, roi_selection, useblit=True)         # công cụ chọn vùng hình chữ nhật từ thư viện matplotlib.widgets
plt.show()                                                      # hiển thị cửa sổ chọn vùng 

roi_h, roi_w = roi_array.shape                                  # kích thước vùng ROI
print(f" - ROI kích thước: {roi_w} x {roi_h} pixels")           # in ra kích thước vùng ROI

# Bước 2: Xác định vị trí vật thể trong từng khung hình
print("- Bắt đầu xác định vị trí vật thể...")

roi_float = roi_array.astype(np.float32)            # chuyển vùng ROI sang dạng float32
roi_float -= roi_float.mean()                       # trừ đi giá trị trung bình để chuẩn hoá

# Duyệt qua từng khung hình để tìm vị trí vật thể
for i, t in enumerate(timestamps):
    frame_rgb = clip.get_frame(t)                                        # lấy khung hình tại thời điểm t
    frame_gray = np.array(Image.fromarray(frame_rgb).convert('L'))       # chuyển sang ảnh xám

    img_float = frame_gray.astype(np.float32)                            # chuyển ảnh sang dạng float32
    img_float -= img_float.mean()                                        # trừ đi giá trị trung bình để chuẩn hoá

    corr = correlate2d(img_float, roi_float, mode='same')                # tính toán ma trận tương quan 2D từ thư viện scipy.signal correlate2d
    y, x = np.unravel_index(np.argmax(corr), corr.shape)                 # tìm toạ độ (x, y) của giá trị lớn nhất trong ma trận tương quan

    x1, y1 = max(0, x - roi_w // 2), max(0, y - roi_h // 2)              # toạ độ góc trên bên trái của vùng phát hiện
    x2, y2 = x1 + roi_w, y1 + roi_h                                      # toạ độ góc dưới bên phải của vùng phát hiện
    positions.append((x, y))                                             # lưu toạ độ tâm vật thể

    # Ảnh màu có khung
    color_pil = Image.fromarray(frame_rgb)                               # chuyển mảng numpy sang đối tượng PIL
    draw_c = ImageDraw.Draw(color_pil)                                   # tạo đối tượng vẽ
    draw_c.rectangle([x1, y1, x2, y2], outline='red', width=3)           # vẽ hình chữ nhật quanh vật thể   
    color_pil.save(os.path.join(color_folder, f"color_{i:05d}.jpg"))     # lưu ảnh màu có khung   

    # Ảnh xám có khung và tâm
    gray_rgb = np.stack([frame_gray] * 3, axis=-1)                       # chuyển ảnh xám sang ảnh RGB để vẽ màu  
    gray_pil = Image.fromarray(gray_rgb)                                 # chuyển mảng numpy sang đối tượng PIL
    draw_g = ImageDraw.Draw(gray_pil)                                    # tạo đối tượng vẽ
    draw_g.rectangle([x1, y1, x2, y2], outline='red', width=2)           # vẽ hình chữ nhật quanh vật thể

    cx, cy = x1 + roi_w // 2, y1 + roi_h // 2                            # toạ độ tâm vật thể
    draw_g.ellipse([cx-4, cy-4, cx+4, cy+4], fill='lime')                # vẽ tâm vật thể 

    try:
        font = ImageFont.truetype("arial.ttf", 14)             # sử dụng font chữ Arial
    except:
        font = ImageFont.load_default()                        # sử dụng font mặc định nếu không tìm thấy Arial
    draw_g.text((10, 10), f"Frame: {i}\nX: {cx}\nY: {cy}", fill='yellow', font=font)    # hiển thị thông tin toạ độ và khung hình

    gray_pil.save(os.path.join(gray_folder, f"gray_{i:05d}.jpg"))                       # lưu ảnh xám có khung và tâm

    print(f" - Đã xử lý {i}/{len(timestamps)} khung hình...")                           # in tiến độ xử lý

clip.close()
print("- Hoàn tất theo dõi.")

# Bước 3: Vẽ biểu đồ quỹ đạo chuyển động
print("- Đang vẽ biểu đồ quỹ đạo...")

x_coords = [x for x, y in positions]                                                       # trích toạ độ x từ danh sách positions
y_coords = [y for x, y in positions]                                                       # trích toạ độ y từ danh sách positions

plt.figure(figsize=(10, 8))                                                                # tạo cửa sổ vẽ
plt.plot(x_coords, y_coords, '-o', markersize=3, linewidth=1)                              # vẽ quỹ đạo chuyển động
plt.title("Quỹ đạo chuyển động của vật thể")
plt.xlabel("Trục X (pixels)")
plt.ylabel("Trục Y (pixels)")
plt.xlim(0, first_frame_gray.shape[1])                                                     # giới hạn trục X theo kích thước ảnh
plt.ylim(first_frame_gray.shape[0], 0)                                                     # giới hạn trục Y theo kích thước ảnh (đảo ngược trục Y)
plt.grid(True, alpha=0.3)                                                        
plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Bắt đầu')       # đánh dấu điểm bắt đầu
plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', label='Kết thúc')      # đánh dấu điểm kết thúc
plt.legend() 
plt.tight_layout()

trajectory_path = os.path.join(output_base, "BieuDo.png")                                  # đường dẫn lưu biểu đồ quỹ đạo 
plt.savefig(trajectory_path, dpi=300)                                                      # lưu biểu đồ quỹ đạo
plt.show()
print(f" - Đã lưu biểu đồ quỹ đạo tại: {trajectory_path}")

# Bước 4: Tạo video đầu ra với khung vị trí vật thể
print("- Đang tạo video đầu ra...")

color_files = [os.path.join(color_folder, f"color_{i:05d}.jpg") for i in range(len(timestamps))]    # danh sách file ảnh màu có khung
output_clip = ImageSequenceClip(color_files, fps=fps)                                               # tạo video từ chuỗi ảnh
output_path = os.path.join(output_folder, "Videodaxuly.mp4")                                        # đường dẫn lưu video đầu ra
output_clip.write_videofile(output_path, codec='libx264', audio=False)                              # lưu video đầu ra
output_clip.close()     

print(f"=== HOÀN THÀNH ===")
print(f"Ảnh xám:  {gray_folder}")
print(f"Ảnh màu:  {color_folder}")
print(f"Video:    {output_path}")
print(f"Biểu đồ:  {trajectory_path}")
