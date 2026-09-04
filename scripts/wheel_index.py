"""Write PEP 503 index pages for the wheels attached to GitHub releases.

Usage: wheel_index.py <owner/repo> <output dir>

One index per CUDA tag: <out>/whl/<tag>/isoext/index.html lists every wheel
whose local version is +<tag>, linking to the release asset. Users install
with `pip install isoext --extra-index-url https://<pages>/whl/<tag>`.
"""

import html
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path


def main():
    repo, out = sys.argv[1], Path(sys.argv[2])
    with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/releases?per_page=100") as r:
        releases = json.load(r)
    wheels = defaultdict(list)
    for release in releases:
        for asset in release["assets"]:
            m = re.match(r"isoext-[^-]+\+(cu\d+)-.*\.whl$", asset["name"])
            if m:
                wheels[m.group(1)].append((asset["name"], asset["browser_download_url"]))
    for tag, files in sorted(wheels.items()):
        index = out / "whl" / tag / "isoext" / "index.html"
        index.parent.mkdir(parents=True, exist_ok=True)
        links = "\n".join(f'<a href="{url}">{html.escape(name)}</a><br>' for name, url in sorted(files))
        index.write_text(f"<!DOCTYPE html>\n<html><body>\n{links}\n</body></html>\n")
        (out / "whl" / tag / "index.html").write_text('<!DOCTYPE html>\n<html><body>\n<a href="isoext/">isoext</a>\n</body></html>\n')
        print(f"{tag}: {len(files)} wheels")


if __name__ == "__main__":
    main()
