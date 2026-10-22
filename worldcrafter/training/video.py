import numpy as np
import torch
import torchvision

resolution_bucket_options = {
    640: [
        (768, 320),
        (768, 384),
        (640, 384),
        (768, 512),
        (576, 448),
        (512, 512),
        (448, 576),
        (512, 768),
        (384, 640),
        (384, 768),
        (320, 768),
    ],
}

length_bucket_options = {
    1: [
        501,
        481,
        461,
        441,
        421,
        401,
        381,
        361,
        341,
        321,
        301,
        281,
        261,
        241,
        221,
        193,
        181,
        161,
        141,
        121,
        101,
        81,
        61,
        41,
        21,
    ],
    2: [193, 177, 161, 156, 145, 133, 129, 121, 113, 109, 97, 85, 81, 73, 65, 61, 49, 37, 25],
}


def find_nearest_resolution_bucket(h, w, resolution=640):
    min_metric = float("inf")
    best_bucket = None
    for bucket_h, bucket_w in resolution_bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket


def find_nearest_length_bucket(length, stride=1):
    buckets = length_bucket_options[stride]
    min_bucket = min(buckets)
    if length < min_bucket:
        return length
    valid_buckets = [bucket for bucket in buckets if bucket <= length]
    return max(valid_buckets)


def read_cut_crop_and_resize(
    video_path, f_prime, h_prime, w_prime, stride=1, start_frame=None, end_frame=None, crop=None
):
    frame_indices = list(range(start_frame, end_frame, stride))
    assert len(frame_indices) == f_prime

    import av
    wanted = set(frame_indices)
    decoded = []
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded.append(frame.to_ndarray(format="rgb24"))
            if index >= frame_indices[-1]:
                break
    if len(decoded) != f_prime:
        raise ValueError(f"Decoded {len(decoded)} of {f_prime} requested frames: {video_path}")
    frames = torch.from_numpy(np.stack(decoded)).float()

    frames = (frames / 127.5) - 1
    video = frames.permute(0, 3, 1, 2)

    s_x, e_x, s_y, e_y = crop
    video = video[:, :, s_y:e_y, s_x:e_x]

    frames, channels, h, w = video.shape
    aspect_ratio_original = h / w
    aspect_ratio_target = h_prime / w_prime

    if aspect_ratio_original >= aspect_ratio_target:
        new_h = int(w * aspect_ratio_target)
        top = (h - new_h) // 2
        bottom = top + new_h
        left = 0
        right = w
    else:
        new_w = int(h / aspect_ratio_target)
        left = (w - new_w) // 2
        right = left + new_w
        top = 0
        bottom = h

    # Crop the video
    cropped_video = video[:, :, top:bottom, left:right]
    # Resize the cropped video
    resized_video = torchvision.transforms.functional.resize(cropped_video, (h_prime, w_prime))
    return resized_video
