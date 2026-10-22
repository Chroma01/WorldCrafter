"""Behavioral checks for the shared dataset-preparation path."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from training.utils.preparation import inside, normalize_record, validate_camera, target_chunks
from training.prepare_data import merge_indices


def fixture():
    row = normalize_record({'sample_id': 'garden', 'video_path': 'videos/garden.mp4',
        'caption': 'A camera moves through a garden.', 'cut': [7, 172],
        'crop': [0, 1920, 0, 1080], 'fps': 24.0, 'stride': 1})
    n = row['bucket_num_frame'] // 33
    grid = np.arange(n)[:, None] * 33 + np.arange(9)[None, :] * 4
    pose = np.tile(np.eye(4), (n, 9, 1, 1))
    pose[..., 0, 3] = grid * .03
    camera = {key: row[key] for key in ('sample_id', 'selected_start_frame', 'selected_end_frame',
        'bucket_num_frame', 'bucket_height', 'bucket_width', 'stride')}
    camera.update(source_video_path=row['video_path'], prompt_raw=row['caption'], camera_num_chunks=n,
        camera_anchor_processed_frame_idx=grid.tolist(), camera_anchor_frame_idx=(7 + grid).tolist(),
        camera_pose_rel_video_c2w=pose.tolist(),
        camera_pose_rel_chunk_c2w=(np.linalg.inv(pose[:, :1]) @ pose).tolist(),
        camera_K_norm=np.tile([.8, .8, .5, .5], (n, 9, 1)).tolist(),
        camera_valid_mask=np.ones((n, 9), dtype=bool).tolist())
    return row, camera


class PreparationTests(unittest.TestCase):
    def test_custom_and_released_record_preserve_frame_alignment(self):
        row, camera = fixture()
        self.assertEqual(row['bucket_num_frame'], 161)
        self.assertEqual(row['selected_end_frame'], 168)
        self.assertEqual(normalize_record(row), row)
        validate_camera(row, camera)
        row['cut'] = [7, 337]
        for key in ('bucket_num_frame', 'bucket_height', 'bucket_width', 'selected_start_frame', 'selected_end_frame'):
            del row[key]
        row['stride'] = 2
        second = normalize_record(row)
        self.assertEqual(second['bucket_num_frame'], 161)
        self.assertEqual(second['selected_end_frame'], 328)

    def test_camera_mismatches_are_rejected(self):
        row, camera = fixture()
        for change in ('frames', 'intrinsics', 'pose', 'caption', 'crop'):
            wrong = copy.deepcopy(camera)
            if change == 'frames': wrong['camera_anchor_frame_idx'][1][2] += 1
            if change == 'intrinsics': wrong['camera_K_norm'][0][0][0] = -1
            if change == 'pose': wrong['camera_pose_rel_chunk_c2w'][1][2][0][3] += .5
            if change == 'caption': wrong['prompt_raw'] = 'Different video'
            if change == 'crop': wrong['crop'] = [0, 640, 0, 384]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_camera(row, wrong)

    def test_published_target_selection_is_preserved(self):
        row, camera = fixture()
        row['valid_chunk_indices'] = [0, 2]
        self.assertEqual(target_chunks(row, camera), [0, 2])
        row['valid_chunk_indices'] = [99]
        with self.assertRaises(ValueError): target_chunks(row, camera)

    def test_invalid_manifest_and_path_escape(self):
        row, _ = fixture()
        for key, value in [('stride', 0), ('caption', ''), ('sample_id', '../escape'), ('crop', [0, 0, 0, 10])]:
            with self.subTest(key=key), self.assertRaises(ValueError): normalize_record(dict(row, **{key: value}))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): inside(Path(tmp), '../outside.mp4')
            with self.assertRaises(ValueError): inside(Path(tmp), '/absolute.mp4')

    def test_video_with_non_utf8_metadata_decodes_unchanged_pixels(self):
        import av
        import torch
        from training.utils.data import read_cut_crop_and_resize
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / 'good.mp4'
            marker = 'WorldCrafter-metadata-marker'
            with av.open(str(good), 'w') as out:
                out.metadata['title'] = marker
                stream = out.add_stream('mpeg4', rate=24)
                stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
                for i in range(2):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 80 + i * 30, dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame): out.mux(packet)
                for packet in stream.encode(): out.mux(packet)
            raw = good.read_bytes()
            self.assertEqual(raw.count(marker.encode()), 1)
            bad = Path(tmp) / 'bad.mp4'
            bad.write_bytes(raw.replace(marker.encode(), bytes([0xed]) + marker.encode()[1:]))
            with self.assertRaises(UnicodeDecodeError): av.open(str(bad))
            args = dict(f_prime=2, h_prime=384, w_prime=640, stride=1, start_frame=0, end_frame=2, crop=[0, 64, 0, 48])
            self.assertTrue(torch.equal(read_cut_crop_and_resize(good, **args), read_cut_crop_and_resize(bad, **args)))
            args['crop'] = [0, 65, 0, 48]
            with self.assertRaises(ValueError): read_cut_crop_and_resize(good, **args)

    def test_merge_requires_every_shard_and_matching_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'features').mkdir()
            part = {'valid_chunk_indices': {'garden.pt': [1, 2]}, 'shard_index': 0, 'num_shards': 2,
                    'preparation': {'seed': 42, 'manifest_sha256': 'one'}}
            first = root / 'valid_chunk_indices.shard-00000-of-00002.json'
            first.write_text(json.dumps(part))
            (root / 'features/garden.pt').touch()
            with self.assertRaises(FileNotFoundError): merge_indices(root, 2)
            second = root / 'valid_chunk_indices.shard-00001-of-00002.json'
            part.update(valid_chunk_indices={}, shard_index=1)
            second.write_text(json.dumps(part))
            merge_indices(root, 2)
            self.assertEqual(json.loads((root / 'valid_chunk_indices.json').read_text())['valid_chunk_indices'], {'garden.pt': [1, 2]})
            part['preparation']['seed'] = 0
            second.write_text(json.dumps(part))
            with self.assertRaises(ValueError): merge_indices(root, 2)


if __name__ == '__main__':
    unittest.main()
