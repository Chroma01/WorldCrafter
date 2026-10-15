import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from demo.service import Manager
from demo.config import EXAMPLES, presets


class DemoCaptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.data = patch("demo.service.DATA", Path(self.directory.name))
        self.data.start()
        self.loop = asyncio.new_event_loop()
        self.manager = Manager(self.loop, mock=True)
        with self.manager.cv:
            self.assertTrue(self.manager.cv.wait_for(lambda: self.manager.ready, 5))

    def tearDown(self):
        self.manager.close()
        self.loop.close()
        self.data.stop()
        self.directory.cleanup()

    def test_caption_runs_on_owner_and_blocks_other_work(self):
        entered, release = threading.Event(), threading.Event()
        owner = []

        def generate(image, style):
            owner.append(threading.get_ident())
            entered.set()
            release.wait(5)
            return {"prompt": "A garden.", "style": style}

        with patch.object(self.manager, "generate_prompt", side_effect=generate):
            future = self.manager.request_caption("image.png", "first-person")
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(ValueError):
                    self.manager.request_caption("other.png", "third-person")
                with self.assertRaises(ValueError):
                    self.manager.create("image.png", "scene", 42, 1, gravity_align=False)
            finally:
                release.set()
            self.assertEqual(future.result(5)["prompt"], "A garden.")
            self.assertEqual(owner, [self.manager.thread.ident])

    def test_active_session_rejects_caption(self):
        self.manager.create("image.png", "scene", 42, 1, gravity_align=False)
        with self.assertRaises(ValueError):
            self.manager.request_caption("image.png", "first-person")

    def test_caption_error_does_not_break_worker(self):
        with patch.object(self.manager, "generate_prompt", side_effect=[RuntimeError("caption failed"), {"prompt": "A garden."}]):
            future = self.manager.request_caption("image.png", "first-person")
            with self.assertRaisesRegex(RuntimeError, "caption failed"):
                future.result(5)
            with self.manager.cv:
                self.assertTrue(self.manager.cv.wait_for(lambda: self.manager.caption_request is None, 5))
            self.assertEqual(self.manager.request_caption("image.png", "third-person").result(5)["prompt"], "A garden.")
            self.assertIsNone(self.manager.error)

    def test_presets_use_shared_case_images_and_prompts(self):
        items = presets()
        cases = {p.name for p in EXAMPLES.iterdir() if (p / "image.png").is_file() and (p / "prompt.txt").is_file()}
        self.assertEqual({p["id"] for p in items}, cases)
        for item in items:
            self.assertEqual(item["image"], f"/inputs/{item['id']}/image.png")
            self.assertEqual(item["prompt"], (EXAMPLES / item["id"] / "prompt.txt").read_text().strip())
