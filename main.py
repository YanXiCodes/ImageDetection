import cv2
import numpy as np

# 初始化颜色字典，包含每种颜色的HSV边界和窗口名称
colors = {
    'green': {
        'lower_bound': np.array([40, 50, 50]),
        'upper_bound': np.array([80, 255, 255]),
        'window': 'Green Color Adjustments'
    },
    # 红色需要两组边界值，因为HSV色彩空间中红色跨越0度
    'red1': {
        'lower_bound': np.array([0, 150, 0]),
        'upper_bound': np.array([10, 255, 255]),
        'window': 'Red Color Adjustments 1'
    },
    'red2': {
        'lower_bound': np.array([160, 150, 0]),
        'upper_bound': np.array([179, 255, 255]),
        'window': 'Red Color Adjustments 2'
    },
    'orange': {
        'lower_bound': np.array([10, 100, 20]),
        'upper_bound': np.array([25, 255, 255]),
        'window': 'Orange Color Adjustments'
    },
    'gray': {
        'lower_bound': np.array([0, 0, 46]),     # 考虑灰色通常饱和度低而亮度不低
        'upper_bound': np.array([179, 50, 220]),  # 调整为较广泛的范围以覆盖不同灰色调
        'window': 'Gray Color Adjustments'
    },
    'cyan': {
        'lower_bound': np.array([85, 50, 50]),    # 青色的HSV范围可能在绿色到蓝色之间
        'upper_bound': np.array([115, 255, 255]),  # 调整范围以覆盖试管帽青色
        'window': 'Cyan Color Adjustments'
    },
    'purple': {
        'lower_bound': np.array([125, 50, 50]),    # 紫色的HSV范围一般比青色更接近蓝色
        'upper_bound': np.array([155, 255, 255]),  # 调整范围以覆盖试管帽紫色
        'window': 'Purple Color Adjustments'
    }
 
}


#试管特征轮廓的面积阈值
min_area_threshold = 500  
max_area_threshold = 1500  

def get_mask_for_color(hsv_image, color_range):
    lower_bound, upper_bound = color_range
    # 在HSV空间中基于颜色范围来创建掩码
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def apply_morphology(mask):
    kernel = np.ones((3,3), np.uint8)  # 可能需要调整
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask

def apply_watershed(image, mask):
    # 对掩码应用膨胀运算以确保区域之间有足够的距离
    sure_bg = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=3)
    
    # 应用距离转换并通过阈值化得到确定的前景区域
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sure_fg_threshold = 0.7 * dist_transform.max()
    _, sure_fg = cv2.threshold(dist_transform, sure_fg_threshold, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 获得未知区域（即可能是前景也可能是背景的区域）
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 使用connectedComponents来标记确定的前景区域
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # 增加所有的标记
    markers[unknown == 255] = 0  # 未知区域标为0

    # 应用分水岭算法
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 0]  # 将找到的边界标为绿色

    return markers

def count_and_mark_tubes(image, markers, min_area_threshold, max_area_threshold):
    tube_count = 0
    # 遍历所有的标记
    for marker in np.unique(markers):
        if marker == 0 or marker == -1:
            continue
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[markers == marker] = 255

        # 查找轮廓
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # 设置面积阈值，仅计算满足条件的轮廓
            print(min_area_threshold,max_area_threshold)
            if area > min_area_threshold and area < max_area_threshold:
                tube_count += 1
                # 在原图上绘制轮廓
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
                # 使用轮廓的重心来放置文本
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(image, f"Tube {tube_count}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return tube_count


# 滑动条的回调函数
def on_trackbar(val):
   pass

# 为每种颜色创建滑动条
for color_name, color_data in colors.items():
    cv2.namedWindow(color_data['window'])
    cv2.createTrackbar('Lower H', color_data['window'], color_data['lower_bound'][0], 179, on_trackbar)
    cv2.createTrackbar('Upper H', color_data['window'], color_data['upper_bound'][0], 179, on_trackbar)
    cv2.createTrackbar('Lower S', color_data['window'], color_data['lower_bound'][1], 255, on_trackbar)
    cv2.createTrackbar('Upper S', color_data['window'], color_data['upper_bound'][1], 255, on_trackbar)
    cv2.createTrackbar('Lower V', color_data['window'], color_data['lower_bound'][2], 255, on_trackbar)
    cv2.createTrackbar('Upper V', color_data['window'], color_data['upper_bound'][2], 255, on_trackbar)

# 创建一个新窗口用于调整面积阈值
cv2.namedWindow('Area Threshold Adjustments')
# 创建最小面积阈值滑动条
cv2.createTrackbar('Min Area', 'Area Threshold Adjustments', min_area_threshold, 5000, on_trackbar)
# 创建最大面积阈值滑动条
cv2.createTrackbar('Max Area', 'Area Threshold Adjustments', max_area_threshold, 5000, on_trackbar)

# 更新滑动条值
def update_trackbar_positions(color_name):
    color_data = colors[color_name]
    color_data['lower_bound'][0] = cv2.getTrackbarPos('Lower H', color_data['window'])
    color_data['upper_bound'][0] = cv2.getTrackbarPos('Upper H', color_data['window'])
    color_data['lower_bound'][1] = cv2.getTrackbarPos('Lower S', color_data['window'])
    color_data['upper_bound'][1] = cv2.getTrackbarPos('Upper S', color_data['window'])
    color_data['lower_bound'][2] = cv2.getTrackbarPos('Lower V', color_data['window'])
    color_data['upper_bound'][2] = cv2.getTrackbarPos('Upper V', color_data['window'])



# 加载图片并转换HSV
image = cv2.imread(r'TestTubeQuantityDetection/resources/testtube_1.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    combined_mask = None
    # 使用原始图像的副本
    image_with_contours = image.copy()
    # 更新最小和最大面积阈值
    min_area_threshold = cv2.getTrackbarPos('Min Area', 'Area Threshold Adjustments')
    max_area_threshold = cv2.getTrackbarPos('Max Area', 'Area Threshold Adjustments')
    for color_name, color_data in colors.items():
        # 更新滑动条的值
        update_trackbar_positions(color_name)
        mask = get_mask_for_color(hsv_image, (color_data['lower_bound'], color_data['upper_bound']))
        
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 应用联合掩码以及其他图像处理步骤
    if combined_mask is not None:
        cleaned_mask = apply_morphology(combined_mask)
        # 获取分水岭算法的标记
        markers = apply_watershed(image_with_contours, cleaned_mask)
        # 标记轮廓并计数
        tube_count = count_and_mark_tubes(image_with_contours, markers,min_area_threshold,max_area_threshold)
        print(min_area_threshold,max_area_threshold)
        # print(f'Detected tubes: {tube_count}')
        # 显示带有标记的原始图像副本
        cv2.imshow('Processed Image', image_with_contours)
        # 显示当前掩码图像
        cv2.imshow('Current Mask', cleaned_mask)

    # 检查用户是否按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()