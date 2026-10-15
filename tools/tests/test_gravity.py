import asyncio
import json
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from demo.camera import Camera
from demo import service
from worldcrafter.camera import Action, action_from_event, build_trajectory, relative_poses
from worldcrafter.cli import parse_args, prepare_camera
from worldcrafter.gravity import align_trajectory, gravity_frame

UP = np.array([.3, -.9, .2]) / np.linalg.norm([.3, -.9, .2])
REPORT = dict(world_up=UP.tolist(), gravity_uncertainty_deg=1., version="test")


class GeometryTests(unittest.TestCase):
    def test_frames_and_yaw_preserve_gravity(self):
        for up in ([0, -1, 0], UP, [-.6, -.8, 0]):
            frame = gravity_frame(up)
            np.testing.assert_allclose(frame @ up, [0, -1, 0], atol=1e-12)
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame), 1.)
            camera = Camera(up)
            for action in [Action(yaw=30), Action(yaw=-15, orbit=True, orbit_radius=2)]:
                _, poses = camera.append(action)
                np.testing.assert_allclose(poses[:, :3, :3] @ up, np.broadcast_to(up, (len(poses), 3)), atol=1e-6)
            np.testing.assert_allclose(poses[0], np.eye(4)[:3], atol=1e-7)

    def test_demo_and_cli_actions_match_including_orbit_and_translation(self):
        events = ['yaw_left30', 'orbit_right15', 'pitch_up15', 'forward1', 'up1', 'orbit_down10']
        expected, records = build_trajectory(events, world_up=UP)
        camera = Camera(UP)
        for event in events:
            local, global_poses = camera.append(action_from_event(event))
        np.testing.assert_allclose(global_poses, expected[:, :3], atol=1e-6)
        for start in range(0, len(expected), 33):
            np.testing.assert_allclose(local[start:start+33], relative_poses(expected[start:start+33]), atol=1e-6)
        np.testing.assert_allclose(records[0]['logical_start_c2w'], np.eye(4), atol=1e-12)
        # Horizontal translation is perpendicular to gravity, elevation is parallel.
        forward = expected[131, :3, 3] - expected[99, :3, 3]
        self.assertAlmostEqual(float(forward @ UP), 0., places=6)
        delta = expected[164, :3, 3] - expected[132, :3, 3]
        np.testing.assert_allclose(delta, UP * 32/33, atol=1e-6)

    def test_external_path_conjugation_and_disabled_level_identity(self):
        path, _ = build_trajectory(['yaw_left30', 'orbit_right15', 'forward1'])
        for value in (path, path[:, :3], path[None], path[None, :, :3]):
            aligned = align_trajectory(value, UP)
            self.assertEqual(aligned.shape, value.shape)
            self.assertEqual(aligned.dtype, value.dtype)
            np.testing.assert_allclose(aligned[..., 0, :, :], value[..., 0, :, :], atol=1e-12)
            np.testing.assert_allclose(align_trajectory(value, [0,-1,0]), value, atol=1e-12)
        frame = gravity_frame(UP)
        aligned = align_trajectory(path, UP)
        np.testing.assert_allclose(aligned[:, :3, 3], path[:, :3, 3] @ frame, atol=1e-12)
        self.assertGreater(np.max(np.abs(aligned - path)), .1)

    def test_invalid_gravity(self):
        for value in ([0,0,0], [0,-2,0], [0,0,1], [float('nan'),0,0]):
            with self.assertRaises(ValueError):
                gravity_frame(value)


class CLITests(unittest.TestCase):
    def test_calibration_failure_preserves_source_and_continues_without_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'image.png'
            Image.new('RGB',(640,384)).save(image)
            path = root/'source.npy'
            source, _ = build_trajectory(['yaw_left30'])
            np.save(path, source)
            for failure in (RuntimeError('model failed'), ImportError('missing dependency'), ValueError('invalid gravity')):
                args = parse_args(['--image-path',str(image),'--camera-path',str(path),'--output-path',str(root/'result.mp4')])
                with patch('worldcrafter.calibration._calibrate_image', side_effect=failure):
                    prepare_camera(args)
                self.assertEqual(args.camera_path,path)
                np.testing.assert_array_equal(np.load(path),source)
                self.assertFalse(args.gravity_calibration['enabled'])
                self.assertEqual(args.gravity_calibration['status'],'fallback')
                self.assertFalse((root/'calibration.json').exists())
                self.assertTrue((root/'result_trajectory/calibration.json').is_file())

    def test_uncertainty_does_not_disable_alignment(self):
        from worldcrafter.calibration import calibrate_image
        with patch('worldcrafter.calibration._calibrate_image', return_value=dict(REPORT, gravity_uncertainty_deg=20.)):
            report = calibrate_image('unused.png')
        self.assertTrue(report['enabled'])
        self.assertNotIn('warning',report)

    def test_switches_inputs_and_saved_effective_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'image.png'
            Image.new('RGB', (640,384)).save(image)
            original = root/'source.npy'
            source, _ = build_trajectory(['yaw_left30'])
            np.save(original, source)
            original_bytes = original.read_bytes()
            for mode in ('i2v', 't2v'):
                for enabled in (True, False):
                    for actions in (True, False):
                        args = parse_args(['--mode', mode, '--output-path', str(root/'result.mp4'),
                            *(['--image-path', str(image)] if mode == 'i2v' else []),
                            *(['--actions','yaw_left30'] if actions else ['--camera-path',str(original)]),
                            *([] if enabled else ['--no-gravity-align'])])
                        with patch('worldcrafter.calibration.calibrate_image', return_value=dict(REPORT)) as calibrate:
                            prepare_camera(args)
                        self.assertEqual(calibrate.call_count, int(enabled and mode == 'i2v'))
                        actual = np.load(args.camera_path)
                        if enabled and mode == 'i2v':
                            self.assertIsNotNone(args.gravity_calibration)
                            self.assertTrue((args.camera_path.parent/'calibration.json').is_file())
                            self.assertGreater(np.max(np.abs(actual[..., :3,:] - source[:, :3])), .01)
                        else:
                            self.assertIsNone(args.gravity_calibration)
                            np.testing.assert_array_equal(actual[..., :3,:], source[:, :3])
                        self.assertEqual(original.read_bytes(), original_bytes)


class ExploreTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_chunk_receives_aligned_local_and_global_poses(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(service, 'DATA', Path(temp)), patch(
            'worldcrafter.calibration.calibrate_image', return_value=REPORT
        ), patch.object(service, 'encode_chunk', side_effect=lambda frames, path: Path(path).write_bytes(b'test')):
            image = Path(temp)/'image.png'
            Image.new('RGB', (640,384)).save(image)
            manager = service.Manager(asyncio.get_running_loop(), mock=True)
            try:
                await self.until(lambda: manager.ready)
                state = manager.create(image, 'office', 42, 1)
                session = manager.sessions[state['session_id']]
                manager.connect(session.id, asyncio.Queue())
                manager.command(session.id, dict(type='key', key='j', down=True, generation=0))
                await self.until(lambda: session.chunks == 1)
                expected = Camera(UP)
                local, global_poses = expected.append(Action(yaw=30, orbit=True, orbit_radius=2))
                np.testing.assert_allclose(session.global_poses[0], global_poses, atol=1e-6)
                np.testing.assert_allclose(session.local_poses[0], local, atol=1e-6)
                saved = json.loads((session.directory/'session.json').read_text())
                self.assertEqual(saved['calibration'], REPORT)
                self.assertTrue(saved['gravity_align'])
            finally:
                manager.close()
                await asyncio.to_thread(manager.thread.join, 5)

    async def test_explore_calibrates_before_any_key_and_reuses_on_restart(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(service, 'DATA', Path(temp)):
            image = Path(temp)/'image.png'
            Image.new('RGB', (640,384)).save(image)
            with patch('worldcrafter.calibration.calibrate_image', return_value=REPORT) as calibrate:
                manager = service.Manager(asyncio.get_running_loop(), mock=True)
                try:
                    await self.until(lambda: manager.ready)
                    self.assertEqual(calibrate.call_count, 0)
                    state = manager.create(image, 'office', 42, 1)
                    session = manager.sessions[state['session_id']]
                    await self.until(lambda: session.calibration is not None)
                    self.assertEqual(session.chunks, 0)
                    self.assertEqual(calibrate.call_count, 1)
                    manager.connect(session.id, asyncio.Queue())
                    manager.command(session.id, {'type':'reset', 'generation':session.generation})
                    await asyncio.sleep(.05)
                    self.assertEqual(calibrate.call_count, 1)
                    self.assertEqual(session.generation, 1)
                    manager.command(session.id, {'type':'stop', 'generation':session.generation})
                    state = manager.create(image, 'office', 42, 1, gravity_align=False)
                    await asyncio.sleep(.05)
                    self.assertEqual(calibrate.call_count, 1)
                    self.assertFalse(state['gravity_align'])
                finally:
                    manager.close()
                    await asyncio.to_thread(manager.thread.join, 5)

    async def test_failure_falls_back_and_still_generates(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(service, 'DATA', Path(temp)), patch(
            'worldcrafter.calibration._calibrate_image', side_effect=RuntimeError('bad gravity')
        ), patch.object(service, 'encode_chunk', side_effect=lambda frames, path: Path(path).write_bytes(b'test')):
            image = Path(temp)/'image.png'
            Image.new('RGB', (640,384)).save(image)
            manager = service.Manager(asyncio.get_running_loop(), mock=True)
            try:
                await self.until(lambda: manager.ready)
                state = manager.create(image, 'office', 42, 1)
                session = manager.sessions[state['session_id']]
                await self.until(lambda: session.calibration is not None)
                self.assertIsNone(session.error)
                self.assertFalse(session.stopped)
                self.assertFalse(session.snapshot()['gravity_aligned'])
                self.assertEqual(session.calibration['status'], 'fallback')
                manager.connect(session.id, asyncio.Queue())
                manager.command(session.id, dict(type='key', key='arrowleft', down=True, generation=0))
                await self.until(lambda: session.chunks == 1)
                expected = Camera()
                local, global_poses = expected.append(Action(yaw=-30))
                np.testing.assert_array_equal(session.global_poses[0], global_poses)
                np.testing.assert_array_equal(session.local_poses[0], local)
            finally:
                manager.close()
                await asyncio.to_thread(manager.thread.join, 5)

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(.01)


class HTTPTests(unittest.TestCase):
    def test_upload_does_not_calibrate_but_explore_does(self):
        from fastapi.testclient import TestClient
        from demo import server
        with tempfile.TemporaryDirectory() as temp, patch.object(server, 'DATA', Path(temp)), patch.object(
            service, 'DATA', Path(temp)
        ), patch.dict(os.environ, {'WORLDCRAFTER_DEMO_MOCK':'1'}), patch(
            'worldcrafter.calibration.calibrate_image', return_value=REPORT
        ) as calibrate, TestClient(server.app) as client:
            buffer = io.BytesIO()
            Image.new('RGB',(800,600)).save(buffer,format='PNG')
            uploaded = client.post('/api/upload',content=buffer.getvalue())
            self.assertEqual(uploaded.status_code,200)
            self.assertEqual(calibrate.call_count,0)
            response = client.post('/api/sessions',json=dict(upload_id=uploaded.json()['upload_id'],prompt='office'))
            self.assertEqual(response.status_code,200)
            sid = response.json()['session_id']
            for _ in range(100):
                status = client.get(f'/api/sessions/{sid}').json()
                if status['calibration'] is not None:
                    break
                time.sleep(.01)
            self.assertEqual(calibrate.call_count,1)
            self.assertTrue(status['gravity_align'])
            self.assertEqual(status['chunks'],0)
            self.assertFalse(status['connected'])
            manager = server.app.state.manager
            manager.command(sid,dict(type='stop',generation=0))
            disabled = client.post('/api/sessions',json=dict(upload_id=uploaded.json()['upload_id'],prompt='office',gravity_align=False))
            self.assertEqual(disabled.status_code,200)
            self.assertFalse(disabled.json()['gravity_align'])
            self.assertEqual(calibrate.call_count,1)
