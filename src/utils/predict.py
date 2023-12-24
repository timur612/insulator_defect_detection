import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def get_predictions(directory_path: str,
                    save_path: str,
                    device: str,
                    ) -> list:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'model')
    model_path = os.path.join(model_dir, 'best700.pt')

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.35,
        device=device
    )

    res = []
    directory_path = os.path.join(current_dir, '..', '..', directory_path)
    files_and_folders = os.listdir(directory_path)
    jpg_files = [file for file in files_and_folders if file.lower().endswith('.jpg')]

    save_path = os.path.join(current_dir, '../..', save_path)
    for item in jpg_files:
        s = f'{directory_path}/{item}'
        result = get_sliced_prediction(s,
                                       detection_model,
                                       slice_height=500,
                                       slice_width=500,
                                       overlap_height_ratio=0.55,
                                       overlap_width_ratio=0.55,
                                       )

        result.export_visuals(save_path, file_name=item)

        size = [result.image_width, result.image_height]
        list_with_coordinates = result.object_prediction_list
        for k, i in enumerate(list_with_coordinates):
            bbox = i.bbox
            x, y, w, h = bbox.to_xywh()
            zz = [x / size[0], y / size[1], w / size[0], h / size[1]]
            zz[0] = zz[0] + zz[2] / 2
            zz[1] = zz[1] + zz[3] / 2
            zz.append(i.score.value)
            zz.append(item[:-4] + '_' + str(k + 1))
            res.append(zz)

    return res
