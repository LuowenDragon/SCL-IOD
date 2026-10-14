import json

def filter_annotations_by_weather_and_time(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    filtered_data = []

    for item in data:
        attrs = item.get("attributes", {})
        weather = attrs.get("weather")
        timeofday = attrs.get("timeofday")

        if weather == "clear" and timeofday == "night":
            filtered_data.append(item)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"符合条件的图片数量: {len(filtered_data)}")
    print(f"已保存到: {output_json_path}")

    return len(filtered_data)

if __name__ == "__main__":
    input_file = "bdd100k_labels_images_train.json"
    output_file = "bdd100k_labels_images_nighttime_clear_train.json"

    count = filter_annotations_by_weather_and_time(input_file, output_file)