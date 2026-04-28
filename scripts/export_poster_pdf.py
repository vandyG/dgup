from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def resolve_chromium() -> str:
    chromium_bin = os.environ.get("CHROMIUM_BIN")
    if chromium_bin:
        return chromium_bin

    for candidate in ("chromium", "chromium-browser", "google-chrome"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise SystemExit(
        "No Chromium executable found. Enter the Nix shell or set CHROMIUM_BIN to a Chromium binary."
    )


def export_pdf(source: Path, output: Path, width_in: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    chromium = resolve_chromium()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--allow-file-access-from-files"],
        )
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        page.goto(source.resolve().as_uri(), wait_until="networkidle")
        page.wait_for_function("() => document.fonts && document.fonts.status === 'loaded'")
        poster = page.locator(".poster")
        poster.wait_for()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_png = Path(temp_file.name)

        try:
            poster.screenshot(path=str(temp_png))
        finally:
            browser.close()

        with Image.open(temp_png) as image:
            dpi = image.width / width_in
            image.convert("RGB").save(output, "PDF", resolution=dpi)
        temp_png.unlink(missing_ok=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export the HTML poster to a 16:9 PDF.")
    parser.add_argument(
        "source",
        nargs="?",
        default=repo_root / "data" / "project_poster_final.html",
        type=Path,
        help="Path to the source HTML poster.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=repo_root / "data" / "project_poster_final.pdf",
        type=Path,
        help="Path to the output PDF.",
    )
    parser.add_argument(
        "--width-in",
        default=16.0,
        type=float,
        help="Target output width in inches. Height is derived from the rendered poster aspect ratio.",
    )
    args = parser.parse_args()

    export_pdf(args.source, args.output, args.width_in)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())