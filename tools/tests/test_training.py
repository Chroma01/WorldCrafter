"""Behavior checks for history modes, feature selection, camera conventions and packaging."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from worldcrafter.training.utils.inputs import sample_sequential_history_mode, build_helios_random_chunk_training_inputs
from worldcrafter.training.utils.loss import prepare_stage1_noise_input
from worldcrafter.training.utils.loss import make_scheduler
from worldcrafter.training.utils.camera import compute_fov_from_fx_xi
from worldcrafter.training.dataset import OSPFeatureDataset
from worldcrafter.training.utils.sampling import eligible_target_chunks

ROOT = Path(__file__).resolve().parents[2]


def feature_fixture():
    n = 3
    poses = torch.eye(4).repeat(n, 9, 1, 1)
    poses[:, :, 0, 3] = torch.arange(n * 9).reshape(n, 9) * .1
    relative = torch.linalg.inv(poses[:, :1]) @ poses
    anchors = torch.arange(n)[:, None] * 33 + torch.arange(0, 33, 4)[None, :]
    return {'vae_latent': torch.randn(n, 16, 9, 48, 80), 'prompt_embed': torch.randn(512, 4096),
            'camera_pose_rel_video_c2w': poses, 'camera_pose_rel_chunk_c2w': relative,
            'camera_K_norm': torch.tensor([.8, .8, .5, .5]).repeat(n, 9, 1),
            'camera_valid_mask': torch.ones(n, 9, dtype=torch.bool),
            'camera_anchor_frame_idx': anchors, 'camera_anchor_processed_frame_idx': anchors}


class TrainingTests(unittest.TestCase):
    def test_sequential_mode_distribution(self):
        generator = torch.Generator().manual_seed(42)
        counts = {key: 0 for key in ('t2v', 'v2v', 'none')}
        for _ in range(20000):
            counts[sample_sequential_history_mode(t2v_ratio=.4, v2v_ratio=.4, generator=generator)] += 1
        for key, expected in [('t2v', .4), ('v2v', .24), ('none', .36)]:
            self.assertAlmostEqual(counts[key] / 20000, expected, delta=.02)

    def test_history_branches_and_noise_keep_source_clean(self):
        target = torch.randn(1, 16, 9, 48, 80)
        history = torch.randn(1, 16, 7, 48, 80)
        x0 = torch.randn(1, 16, 1, 48, 80)
        original = history.clone()
        config = json.loads((ROOT / 'worldcrafter/training/utils/reference_noise.json').read_text())
        for mode in ('t2v', 'v2v', 'none'):
            clean = build_helios_random_chunk_training_inputs(target, history, x0, forced_drop_mode=mode, dtype=torch.float32)
            if mode == 't2v':
                self.assertEqual(torch.count_nonzero(clean['latents_history_short']), 0)
            if mode in ('t2v', 'v2v'):
                self.assertEqual(torch.count_nonzero(clean['latents_history_long']), 0)
            random.seed(43); torch.manual_seed(43)
            outputs = prepare_stage1_noise_input(SimpleNamespace(training_config=SimpleNamespace(**config)),
                clean['input_latents'], make_scheduler(), latents_history_short=clean['latents_history_short'],
                latents_history_mid=clean['latents_history_mid'], latents_history_long=clean['latents_history_long'])
            self.assertTrue(torch.isfinite(outputs[0][0]).all())
            self.assertTrue(torch.equal(history, original))

    def test_dataset_preserves_history_and_native_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature = feature_fixture()
            torch.save(feature, root / 'sample.pt')
            (root / 'valid.json').write_text(json.dumps({'valid_chunk_indices': {'sample.pt': [2]}}))
            data = OSPFeatureDataset(root, feature_files=[str(root / 'sample.pt')],
                valid_chunk_indices_json=root / 'valid.json', dataset_source_name='osp_istock_v2', intrinsics_contract='native')
            sample = data[0]
            self.assertEqual(sample['target_chunk_index'], 2)
            self.assertTrue(torch.equal(sample['input_latents'], feature['vae_latent'][2].bfloat16()))
            self.assertTrue(torch.equal(sample['helios_history_tail_latents'], feature['vae_latent'][1, :, -3:].bfloat16()))
            self.assertAlmostEqual(float(sample['x_fov']), float(compute_fov_from_fx_xi(.8 * 640, 0, 640)), places=4)
            self.assertEqual(tuple(sample['lagernvs_source_latents'].shape), (16, 9, 48, 80))
            self.assertTrue(torch.equal(data[0]['lagernvs_source_latents'], sample['lagernvs_source_latents']))
            self.assertEqual(eligible_target_chunks(feature, [0, 1, 2]), (1, 2))
            feature['camera_valid_mask'][:2] = False
            self.assertEqual(eligible_target_chunks(feature, [2]), ())

    def test_pose_conversion_and_intrinsics(self):
        spec = importlib.util.spec_from_file_location('prepare_features', ROOT / 'tools/data/prepare_features.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        w2c = np.eye(4)[None].repeat(2, 0)
        w2c[1, 0, 3] = -2
        c2w = module.convert_extrinsics_to_c2w(w2c[:, :3])
        self.assertAlmostEqual(c2w[1, 0, 3], 2)
        k = np.array([[[252, 0, 252], [0, 154, 154], [0, 0, 1]]])
        np.testing.assert_allclose(module.intrinsics_to_fxfycxcy_norm(k, 504, 308), [[.5, .5, .5, .5]])


if __name__ == '__main__':
    unittest.main()
