import unittest

from demo.penguin_streamlit_app import main as demo_main


class StreamlitEntrypointTests(unittest.TestCase):
    def test_root_streamlit_entrypoint_reuses_demo_main(self):
        import streamlit_app

        self.assertIs(streamlit_app.main, demo_main)


if __name__ == "__main__":
    unittest.main()
