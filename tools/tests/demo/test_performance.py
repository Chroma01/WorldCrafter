import os
import sys
import unittest
from unittest.mock import patch

from demo.__main__ import main
from demo.multigpu.geometry_cache import GeometryCache


class GeometryCacheTests(unittest.TestCase):
    def test_geometry_changes_and_session_restart(self):
        cache = GeometryCache()
        cache.begin((0, 0, 540))
        self.assertEqual(cache.get('camera', lambda: 'forward-pose'), 'forward-pose')
        self.assertEqual(cache.get('camera', lambda: self.fail('same stage recomputed')), 'forward-pose')
        cache.begin((0, 1, 2160))
        self.assertEqual(cache.get('camera', lambda: 'higher-resolution'), 'higher-resolution')
        cache.begin((1, 0, 540))
        self.assertEqual(cache.get('camera', lambda: 'turned-pose'), 'turned-pose')
        self.assertEqual(cache.stats(), dict(hits=1, misses=3, entries=1))
        cache.reset()
        cache.begin((0, 0, 540))
        self.assertEqual(cache.get('camera', lambda: 'new-session-pose'), 'new-session-pose')
        self.assertEqual(cache.stats(), dict(hits=0, misses=1, entries=1))


class PerformanceSelectionTests(unittest.TestCase):
    def test_four_and_eight_gpu_defaults_and_physical_mapping(self):
        for count in (4, 8):
            visible = ','.join(str(i+8) for i in range(count))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}), \
                 patch.object(sys, 'argv', ['demo', '--devices', ','.join(map(str, range(count)))]), \
                 patch('demo.__main__.run_distributed', return_value=0) as launch:
                with self.assertRaises(SystemExit) as done:
                    main()
                self.assertEqual(done.exception.code, 0)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], visible)
                self.assertEqual(os.environ['WORLDCRAFTER_VAE_MODE'], f'pipeline{count}')
                self.assertEqual(os.environ['WORLDCRAFTER_DEMO_OPTIMIZED'], '1')
                self.assertIn(str(count), launch.call_args.args[0])

    def test_native_profile_and_no_compile_are_explicit(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3'}), \
             patch.object(sys, 'argv', ['demo', '--devices', '0,1,2,3', '--performance-profile', 'native', '--no-compile']), \
             patch('demo.__main__.run_distributed', return_value=0):
            with self.assertRaises(SystemExit):
                main()
            self.assertEqual(os.environ['WORLDCRAFTER_DEMO_OPTIMIZED'], '0')
            self.assertEqual(os.environ['WORLDCRAFTER_DEMO_COMPILE'], '0')


if __name__ == '__main__':
    unittest.main()
