import io
import time
from pathlib import Path
import pytest
from PIL import Image

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    pytest.skip("Playwright not installed", allow_module_level=True)


SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


def test_browser_routes_and_visuals():
    """Verify all 4 routes at 1440x900 and 1024 wide, assert zero console errors and only 127.0.0.1 requests."""
    console_errors = []
    external_requests = []

    with sync_playwright() as p:
        # Launch edge / chromium channel
        browser = p.chromium.launch(channel="msedge", headless=True)
        
        # Test both viewports
        viewports = [
            ("1440x900", {"width": 1440, "height": 900}),
            ("1024x768", {"width": 1024, "height": 768}),
        ]

        # Generate a sample bird photo for Try-it upload
        sample_img = Image.new("RGB", (300, 200), color=(180, 50, 40))
        sample_path = SCREENSHOT_DIR / "sample_bird.jpg"
        sample_img.save(sample_path, format="JPEG")

        for vp_name, vp in viewports:
            context = browser.new_context(viewport=vp)
            page = context.new_page()

            # Event listeners
            def on_console(msg):
                if msg.type == "error":
                    console_errors.append(f"[{vp_name}] Console error: {msg.text}")

            def on_request(req):
                url = req.url
                if not (url.startswith("http://127.0.0.1:8000") or url.startswith("data:") or url.startswith("blob:")):
                    external_requests.append(url)

            page.on("console", on_console)
            page.on("request", on_request)

            # Route 1: #compare
            page.goto("http://127.0.0.1:8000/#compare")
            page.wait_for_selector("#kpi-accuracy")
            page.wait_for_timeout(1000)  # Wait for chart render
            page.screenshot(path=str(SCREENSHOT_DIR / f"compare_{vp_name}.png"), full_page=True)

            # Assert KPI values are present
            acc_orig = page.locator("#kpi-acc-orig").text_content()
            assert acc_orig != "—", "Accuracy KPI should be populated"

            # Assert SVGs rendered
            assert page.locator("#chart-accuracy-curve svg").count() > 0, "Curve chart SVG missing"
            assert page.locator("#chart-redundancy-bars svg").count() > 0, "Redundancy bars SVG missing"

            # Route 2: #try
            page.goto("http://127.0.0.1:8000/#try")
            page.wait_for_selector("#image-drop-zone")
            page.wait_for_timeout(500)
            
            # Upload sample image
            page.set_input_files("#file-upload-input", str(sample_path))
            # Wait for prediction response
            page.wait_for_selector("#agreement-pill", state="visible", timeout=20000)
            page.wait_for_timeout(1000)
            page.screenshot(path=str(SCREENSHOT_DIR / f"try_{vp_name}.png"), full_page=True)

            # Route 3: #concepts
            page.goto("http://127.0.0.1:8000/#concepts")
            page.wait_for_selector("#concepts-col-orig")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(SCREENSHOT_DIR / f"concepts_{vp_name}.png"), full_page=True)

            # Route 4: #about
            page.goto("http://127.0.0.1:8000/#about")
            page.wait_for_selector("#table-about-models tbody tr")
            page.wait_for_timeout(500)
            page.screenshot(path=str(SCREENSHOT_DIR / f"about_{vp_name}.png"), full_page=True)

            context.close()

        browser.close()

    # Final assertions
    print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
    print(f"Console errors: {console_errors}")
    print(f"External requests: {external_requests}")

    assert len(console_errors) == 0, f"Found console errors: {console_errors}"
    assert len(external_requests) == 0, f"Found external requests: {external_requests}"
