import os
import cv2
import numpy as np
import zipfile
import uuid
from flask import Flask, render_template, request, send_file, jsonify, url_for
from werkzeug.utils import secure_filename
import pandas as pd


hsv_ranges = [
    ((0, 100, 100), (8, 255, 255)),
    ((10, 100, 100), (25, 255, 255)),
    ((35, 100, 100), (50, 255, 255)),
    ((50, 50, 150), (70, 255, 255)),
    ((165, 100, 80), (179, 255, 200)),
    ((0, 50, 100), (10, 255, 255)),
    ((140, 80, 80), (160, 255, 255)),
    ((0, 30, 100), (10, 120, 220)),
    ((0, 30, 120), (10, 120, 255)),
    ((0, 100, 30), (10, 255, 100)),
    ((0, 10, 200), (179, 40, 255)),
    ((60, 30, 180), (90, 120, 255)),
    ((80, 80, 100), (100, 255, 255)),
    ((140, 50, 80), (160, 150, 200)),
    ((130, 30, 180), (160, 80, 255)),
    ((160, 80, 150), (179, 255, 255)),
    ((50, 20, 200), (70, 80, 255)),
    ((170, 80, 50), (179, 255, 150)),
    ((0, 0, 80), (179, 30, 140)),
    ((0, 0, 100), (179, 30, 160)),
    ((0, 0, 200), (179, 20, 255)),
    ((0, 0, 200), (179, 20, 255)),
    ((15, 50, 80), (30, 150, 200)),
    ((10, 50, 40), (30, 150, 120)),
    ((50, 30, 100), (70, 100, 200)),
    ((145, 100, 100), (170, 255, 255)),
    ((130, 30, 200), (160, 80, 255)),
    ((140, 80, 150), (160, 255, 255)),
    ((140, 100, 100), (160, 255, 255)),
    ((190, 80, 150), (210, 255, 255)),
    ((150, 80, 150), (170, 255, 255)),
    ((70, 80, 20), (90, 255, 100)),
    ((190, 50, 80), (220, 150, 200)),
    ((140, 100, 100), (160, 255, 255)),
    ((50, 30, 100), (70, 100, 200)),
    ((20, 30, 180), (40, 100, 255)),
    ((15, 50, 100), (30, 150, 200)),
    ((35, 30, 180), (55, 100, 255)),
    ((60, 80, 100), (90, 255, 255)),
    ((60, 80, 150), (90, 200, 255)),
    ((40, 80, 30), (80, 255, 150)),
    ((90, 10, 200), (120, 50, 255)),
]

zone_info = [
    ("Жилые зоны", "Ж-1 Зона застройки многоэтажными жилыми домами"),
    ("Жилые зоны", "Ж-2 Зона застройки среднеэтажными жилыми домами"),
    ("Жилые зоны", "Ж-3 Зона застройки малоэтажными жилыми домами"),
    ("Жилые зоны", "Ж-4 Зона застройки индивидуальными жилыми домами"),
    ("Общественно-деловые зоны", "О-1 Зона размещения объектов делового, общественного и коммерческого назначения"),
    ("Общественно-деловые зоны", "О-2 Зона размещения объектов социального назначения"),
    ("Общественно-деловые зоны", "О-3 Зона размещения объектов физической культуры и спорта (спортивных сооружений)"),
    ("Общественно-деловые зоны", "О-3 Зона объектов образования"),
    ("Общественно-деловые зоны", "О-4 Зона объектов здравоохранения"),
    ("Общественно-деловые зоны", "О-6 Зона исторической застройки"),
    ("Общественно-жилая зона", "ОЖ Зона общественно-жилого назначения"),
    ("Зоны рекреационного назначения",
     "Р-1 Зона зеленых насаждений общего пользования (парков, скверов, садов, бульваров)"),
    ("Зоны рекреационного назначения", "Р-2 Зона размещения объектов отдыха и туризма (зона объектов рекреации)"),
    ("Зоны рекреационного назначения",
     "Р-3 Зона размещения объектов отдыха и туризма (зона объектов рекреации) вне границ населенных пунктов"),
    ("Зоны рекреационного назначения", "Р-3 Зона природных ландшафтов"),
    ("Зоны рекреационного назначения", "Р-4 Зона городских лесов"),
    ("Зоны рекреационного назначения", "Р-5 Зона пляжа"),
    ("Курортная зона", "КЗ Курортная зона"),
    ("Производственные и коммунально-складские зоны",
     "П-1 Зона размещения производственных объектов I, II класса опасности"),
    ("Производственные и коммунально-складские зоны",
     "П-2 Зона размещения производственных объектов III класса опасности"),
    ("Производственные и коммунально-складские зоны",
     "П-3 Зона размещения производственных объектов IV, V классов опасности"),
    ("Производственные и коммунально-складские зоны", "П-3 Научно-производственная зона (Ладушкин)"),
    ("Производственные и коммунально-складские зоны", "П-4 Коммунально-складская зона"),
    ("Производственные и коммунально-складские зоны",
     "П-5 Зона размещения производственных и коммунально-складских предприятий, расположенных вне границ населенных пунктов"),
    ("Производственные и коммунально-складские зоны",
     "П-5 Зона размещения коммунальных объектов и инженерной инфраструктуры (Янтарный)"),
    ("Зона инженерной инфраструктуры", "И Зона инженерной инфраструктуры"),
    ("Зоны транспортной инфраструктуры", "Т-1 Зона размещения объектов железнодорожного транспорта"),
    ("Зоны транспортной инфраструктуры", "Т-2 Зона размещения объектов автомобильного транспорта"),
    ("Зоны транспортной инфраструктуры", "Т-3 Зона размещения объектов воздушного транспорта"),
    ("Зоны транспортной инфраструктуры", "Т-3 Зона транспорта и коммерческого использования (Ладушкин)"),
    ("Зоны транспортной инфраструктуры", "Т-4 Зона размещения объектов водного транспорта"),
    ("Зоны специального назначения", "Сп-1 Зона специального назначения, связанная с захоронениями"),
    ("Зоны специального назначения",
     "Сп-2 Зона специального назначения, связанная с размещением государственных объектов"),
    ("Зоны специального назначения", "Сп-3 Зона объектов обращения с отходами"),
    ("Зоны специального назначения", "Сп-4 Зона озелененных территорий специального назначения"),
    ("Зоны сельскохозяйственного использования",
     "Сх-1 Зона сельскохозяйственных угодий в составе земель сельскохозяйственного назначения"),
    ("Зоны сельскохозяйственного использования",
     "Сх-2 Зона, занятая объектами сельскохозяйственного назначения и предназначенная для ведения сельского хозяйства, садоводства, личного подсобного хозяйства, развития объектов сельскохозяйственного назначения из земель сельскохозяйственного назначения"),
    ("Зоны сельскохозяйственного использования",
     "Сх-3 Зона сельскохозяйственного использования из земель населенных пунктов"),
    ("Зоны сельскохозяйственного использования",
     "Сх-4 Зона садоводческих или огороднических некоммерческих товариществ"),
    ("Зоны сельскохозяйственного использования",
     "Сх-5 Зона садоводческих или огороднических некоммерческих товариществ вне границ населенных пунктов"),
    ("Земли лесного фонда", "Земли лесного фонда"),
    ("Земли водного фонда", "Земли водного фонда"),
]


def save_contours_to_excel(list_cont, list_x_y, output_path):
    rows = []
    for i in range(len(zone_info)):
        category, zone_name = zone_info[i]
        for cnt, (cx, cy) in zip(list_cont[i], list_x_y[i]):
            area = cv2.contourArea(cnt)
            rows.append({
                "Координаты(x,y)": f"({cx}, {cy})",
                "Цвет": zone_name,
                "Площадь": area,
                "Тип местности": category
            })
    df = pd.DataFrame(rows, columns=["Координаты(x,y)", "Цвет", "Площадь", "Тип местности"])
    df.to_excel(output_path, index=False, sheet_name="Лист1")


def resize_image_for_display(img, max_display_width=1200, max_display_height=800):
    height, width = img.shape[:2]
    scale_x = max_display_width / width
    scale_y = max_display_height / height
    scale = min(scale_x, scale_y, 1.0)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    else:
        return img, 1.0


def extract_largest_region_with_resize(image_path, output_path, max_display_size=1200, margin=100):
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, None, None
    original_height, original_width = original_img.shape[:2]
    display_img, scale = resize_image_for_display(original_img, max_display_size, max_display_size)
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 248, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    if scale != 1.0:
        largest_contour = (largest_contour / scale).astype(np.int32)
    mask = np.zeros((original_height, original_width), dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(original_width - x, w + 2 * margin)
    h = min(original_height - y, h + 2 * margin)
    result = original_img[y:y + h, x:x + w]
    cv2.imwrite(output_path, result)
    crop_info = {'x': x, 'y': y, 'w': w, 'h': h}
    return result, crop_info, original_img


def is_closed_contour(contour, tolerance=10):
    if len(contour) < 3: return False
    first_point = contour[0][0]
    last_point = contour[-1][0]
    distance = np.sqrt((first_point[0] - last_point[0]) ** 2 + (first_point[1] - last_point[1]) ** 2)
    return distance <= tolerance


def is_line_like_contour(contour, aspect_ratio_threshold=8, solidity_threshold=0.2):
    x, y, w, h = cv2.boundingRect(contour)
    if w < 5 or h < 5: return False
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
    area = cv2.contourArea(contour)
    rect_area = w * h
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    extent = area / rect_area if rect_area > 0 else 0
    return (aspect_ratio > aspect_ratio_threshold or solidity < solidity_threshold or extent < 0.3)


def filter_contours(contours, min_area=50, max_area=15000):
    filtered = []
    for cnt in contours:
        if not is_closed_contour(cnt): continue
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area: continue
        filtered.append(cnt)
    return filtered


def place(image, list_x_y, list_cont, scale=1.0):
    for i in range(len(hsv_ranges)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_ranges[i][0], hsv_ranges[i][1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_contours(contours)
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                list_x_y[i].append((cx, cy))
                list_cont[i].append(cnt)


def scale_contour_to_original(contour, scale, crop_info=None):
    scaled = contour.copy()
    if scale != 1.0:
        for point in scaled:
            point[0][0] = int(point[0][0] / scale)
            point[0][1] = int(point[0][1] / scale)
    if crop_info is not None:
        for point in scaled:
            point[0][0] += crop_info['x']
            point[0][1] += crop_info['y']
    return scaled


def contour_to_svg_path(contour):
    if len(contour) == 0:
        return ""
    path_data = []
    for i, point in enumerate(contour):
        x, y = point[0]
        if i == 0:
            path_data.append(f"M {x} {y}")
        else:
            path_data.append(f"L {x} {y}")
    path_data.append("Z")
    return " ".join(path_data)


def find_similar_contours(contours1, contours2, tolerance_percent=10, image_shape=None):
    similar, diff1, diff2 = [], [], []
    max_dist = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2) * tolerance_percent / 100 if image_shape else 50
    used = set()
    for i, c1 in enumerate(contours1):
        M1 = cv2.moments(c1)
        if M1["m00"] == 0: continue
        cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
        found = False
        for j, c2 in enumerate(contours2):
            if j in used: continue
            M2 = cv2.moments(c2)
            if M2["m00"] == 0: continue
            cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
            if np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) <= max_dist:
                similar.append((c1, c2))
                used.add(j)
                found = True
                break
        if not found:
            diff1.append(c1)
    for j, c2 in enumerate(contours2):
        if j not in used:
            diff2.append(c2)
    return similar, diff1, diff2


def create_highlighted_image_on_original(base_img, changed_contours, crop_info=None, scale=1.0):
    result = base_img.copy()
    for cnt in changed_contours:
        orig_cnt = scale_contour_to_original(cnt, scale, crop_info)
        cv2.drawContours(result, [orig_cnt], -1, (255, 0, 0), -1)
        cv2.drawContours(result, [orig_cnt], -1, (255, 255, 0), 2)
    return result


def compare_all_categories(list_cont1, list_cont2, image_shape):
    all_changed = []
    for i in range(len(list_cont1)):
        c1, c2 = list_cont1[i], list_cont2[i]
        _, d1, d2 = find_similar_contours(c1, c2, tolerance_percent=10, image_shape=image_shape)
        all_changed.extend(d1)
        all_changed.extend(d2)
    return all_changed


def read_change_records_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    records = []
    for _, row in df.iterrows():
        try:
            coord_str = row["Координаты(x,y)"]
            x, y = map(int, coord_str.strip("()").split(","))
            records.append({
                "x": x,
                "y": y,
                "was": row["Было"],
                "became": row["Стало"],
                "type_was": row["Тип местности (было)"],
                "type_became": row["Тип местности (стало)"]
            })
        except Exception as e:
            print(f"[WARN] Ошибка при чтении строки: {e}")
    return records


def load_compatibility_matrix():
    df = pd.read_excel('Совместимость_зон_Калининград_географ.xlsx', index_col=0, header=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

COMPATIBILITY_MATRIX = load_compatibility_matrix()

def process_zone_maps(img1_path, img2_path, output_dir):
    print(f"\n[DEBUG] Начало обработки карт")
    print(f"[DEBUG] Изображение 1: {img1_path}")
    print(f"[DEBUG] Изображение 2: {img2_path}")
    print(f"[DEBUG] Папка вывода: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    list_cont = [[] for _ in range(42)]
    list_x_y = [[] for _ in range(42)]
    list_cont_2 = [[] for _ in range(42)]
    list_x_y_2 = [[] for _ in range(42)]

    res_1, crop_info_1, orig_1 = extract_largest_region_with_resize(img1_path, os.path.join(output_dir, "cropped1.jpg"), margin=450)
    res_2, crop_info_2, orig_2 = extract_largest_region_with_resize(img2_path, os.path.join(output_dir, "cropped2.jpg"), margin=450)

    if res_1 is None or res_2 is None:
        raise ValueError("Не удалось обработать одно из изображений")

    h1, w1 = res_1.shape[:2]
    h2, w2 = res_2.shape[:2]
    if h1 != h2 or w1 != w2:
        if h1 * w1 > h2 * w2:
            res_1 = cv2.resize(res_1, (w2, h2))
            print(f"[DEBUG] Изображение 1 изменено до {w2}x{h2}")
        else:
            res_2 = cv2.resize(res_2, (w1, h1))
            print(f"[DEBUG] Изображение 2 изменено до {w1}x{h1}")

    proc1, scale1 = resize_image_for_display(res_1, 1200, 5000)
    proc2, scale2 = resize_image_for_display(res_2, 1200, 5000)

    place(proc1, list_x_y, list_cont, scale1)
    place(proc2, list_x_y_2, list_cont_2, scale2)

    all1, all2 = [], []
    for i in range(42):
        for cnt in list_cont[i]:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_res = int(cx / scale1)
                cy_res = int(cy / scale1)
                cx_orig = cx_res + crop_info_1['x']
                cy_orig = cy_res + crop_info_1['y']
                all1.append((cx_orig, cy_orig, cnt, i))
        for cnt in list_cont_2[i]:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_res = int(cx / scale2)
                cy_res = int(cy / scale2)
                cx_orig = cx_res + crop_info_2['x']
                cy_orig = cy_res + crop_info_2['y']
                all2.append((cx_orig, cy_orig, cnt, i))

    h_res, w_res = res_1.shape[:2]
    max_dist = np.sqrt(h_res**2 + w_res**2) * 0.1
    used2 = set()
    change_records = []

    for cx1, cy1, cnt1, cat1 in all1:
        best_j = -1
        best_d = float('inf')
        for j, (cx2, cy2, cnt2, cat2) in enumerate(all2):
            if j in used2: continue
            d = np.hypot(cx1 - cx2, cy1 - cy2)
            if d <= max_dist and d < best_d:
                best_d = d
                best_j = j
        if best_j != -1:
            _, _, _, cat2 = all2[best_j]
            if cat1 != cat2:
                change_records.append({
                    "cx": cx1, "cy": cy1, "area": cv2.contourArea(cnt1),
                    "cat_was": cat1, "cat_became": cat2,
                    "contour": cnt1
                })
            used2.add(best_j)
        else:
            change_records.append({
                "cx": cx1, "cy": cy1, "area": cv2.contourArea(cnt1),
                "cat_was": cat1, "cat_became": None,
                "contour": cnt1
            })

    for j, (cx2, cy2, cnt2, cat2) in enumerate(all2):
        if j not in used2:
            change_records.append({
                "cx": cx2, "cy": cy2, "area": cv2.contourArea(cnt2),
                "cat_was": None, "cat_became": cat2,
                "contour": cnt2
            })

    invalid_changes = 0
    total_real_changes = 0

    for r in change_records:
        was = zone_info[r["cat_was"]][1] if r["cat_was"] is not None else None
        became = zone_info[r["cat_became"]][1] if r["cat_became"] is not None else None

        if was is None or became is None:
            continue

        if was == became:
            continue

        total_real_changes += 1

        was_key = was.split()[0]
        became_key = became.split()[0]

        def normalize_key(key):
            if key.startswith("О-3"):
                return "О-3_спорт"
            elif key.startswith("Р-3"):
                return "Р-3_вне_НП"
            elif key.startswith("П-3"):
                return "П-3_опасность"
            elif key.startswith("Т-3"):
                return "Т-3_авиа"
            elif key.startswith("П-5"):
                return "П-5_вне_НП"
            elif key == "Земли лесного фонда":
                return "Лес"
            elif key == "Земли водного фонда":
                return "Вода"
            elif key == "КЗ Курортная зона":
                return "КЗ"
            elif key.startswith("Сх-"):
                return key
            elif key.startswith("Сп-"):
                return key
            elif key.startswith("ОЖ"):
                return "ОЖ"
            else:
                return key

        was_norm = normalize_key(was_key)
        became_norm = normalize_key(became_key)

        try:
            if was_norm in COMPATIBILITY_MATRIX.index and became_norm in COMPATIBILITY_MATRIX.columns:
                allowed = COMPATIBILITY_MATRIX.loc[was_norm, became_norm]
                if allowed == 0:
                    invalid_changes += 1
            else:
                invalid_changes += 1
        except Exception as e:
            print(f"[WARN] Ошибка при проверке совместимости: {was_norm} → {became_norm}, ошибка: {e}")
            invalid_changes += 1

    if total_real_changes > 0:
        violation_percent = (invalid_changes / total_real_changes) * 100
        correctness_percent = 100.0 - violation_percent
    else:
        correctness_percent = 100.0

    similarity_percent = round(correctness_percent, 1)
    similarity_percent = max(0.0, min(100.0, similarity_percent))

    all_changed_contours = [r["contour"] for r in change_records]
    full_result_path = os.path.join(output_dir, "changed_areas_on_original.jpg")
    highlighted = create_highlighted_image_on_original(orig_1, all_changed_contours, crop_info_1, scale1)
    cv2.imwrite(full_result_path, highlighted)

    excel_path = os.path.join(output_dir, "данные_по_зонам.xlsx")
    rows = []
    for r in change_records:
        was = zone_info[r["cat_was"]][1] if r["cat_was"] is not None else "Отсутствует"
        became = zone_info[r["cat_became"]][1] if r["cat_became"] is not None else "Отсутствует"
        type_was = zone_info[r["cat_was"]][0] if r["cat_was"] is not None else ""
        type_became = zone_info[r["cat_became"]][0] if r["cat_became"] is not None else ""
        rows.append({
            "Координаты(x,y)": f"({r['cx']}, {r['cy']})",
            "Было": was,
            "Стало": became,
            "Площадь": r["area"],
            "Тип местности (было)": type_was,
            "Тип местности (стало)": type_became
        })
    df = pd.DataFrame(rows, columns=[
        "Координаты(x,y)",
        "Было",
        "Стало",
        "Площадь",
        "Тип местности (было)",
        "Тип местности (стало)"
    ])
    df.to_excel(excel_path, index=False, sheet_name="Лист1")

    preview_img = cv2.imread(full_result_path)
    preview_resized, _ = resize_image_for_display(preview_img, 800, 600)
    preview_path = os.path.join(output_dir, "preview.jpg")
    cv2.imwrite(preview_path, preview_resized)

    print(f"[DEBUG] Корректность изменений: {similarity_percent:.1f}% (нарушений: {invalid_changes} из {total_real_changes or '0'} изменений)")
    print(f"[DEBUG] Полное изображение: {full_result_path}")
    print(f"[DEBUG] Превью: {preview_path}")
    print(f"[DEBUG] Excel: {excel_path}\n")

    return full_result_path, preview_path, excel_path, round(similarity_percent, 1)


app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150 MB per file

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    print("\n[INFO] Получен запрос на сравнение изображений")

    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if not file1 or not file2:
        print("[ERROR] Один или оба файла не загружены")
        return jsonify({'error': 'Загрузите оба файла!'}), 400
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        print(f"[ERROR] Неподдерживаемый формат: {file1.filename}, {file2.filename}")
        return jsonify({'error': 'Поддерживаются только JPG, PNG, BMP.'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    file1.save(path1)
    file2.save(path2)
    print(f"[INFO] Файлы сохранены: {path1}, {path2}")

    job_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['RESULT_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Создана рабочая директория: {output_dir}")

    try:
        full_img, preview_img, excel_file, similarity = process_zone_maps(path1, path2, output_dir)

        zip_path = os.path.join(output_dir, "результат_анализа_карт.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(full_img, arcname="изменения_на_оригинале.jpg")
            zf.write(excel_file, arcname="данные_по_зонам.xlsx")
        print(f"[INFO] Архив создан: {zip_path}")

        full_url = url_for('static', filename=f'results/{job_id}/changed_areas_on_original.jpg')
        preview_url = url_for('static', filename=f'results/{job_id}/preview.jpg')
        download_url = url_for('download_result', job_id=job_id)

        print(f"[INFO] URL полного изображения: {full_url}")
        print(f"[INFO] URL превью: {preview_url}")
        print(f"[INFO] URL скачивания: {download_url}")

        return jsonify({
            'success': True,
            'similarity': similarity,
            'result_url': preview_url,
            'full_image_url': full_url,
            'download_url': download_url,
            'job_id': job_id  # <-- добавлено!
        })

    except Exception as e:
        print(f"[CRITICAL] Ошибка при обработке: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500


@app.route('/download/<job_id>')
def download_result(job_id):
    zip_path = os.path.join(app.config['RESULT_FOLDER'], job_id, "результат_анализа_карт.zip")
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name="результат_анализа_карт.zip")
    return "Файл не найден", 404


@app.route('/view_with_markers/<job_id>')
def view_with_markers(job_id):
    output_dir = os.path.join(app.config['RESULT_FOLDER'], job_id)
    excel_path = os.path.join(output_dir, "данные_по_зонам.xlsx")
    full_img_path = os.path.join(output_dir, "changed_areas_on_original.jpg")

    if not os.path.exists(excel_path) or not os.path.exists(full_img_path):
        return "Файлы не найдены", 404

    records = read_change_records_from_excel(excel_path)
    img_url = url_for('static', filename=f'results/{job_id}/changed_areas_on_original.jpg')

    return render_template('markers_view.html', image_url=img_url, records=records)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)