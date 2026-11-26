"""Screenshots of the workstation for the UI guide.

usage: python tools/shoot_ui.py            # writes docs/img/*.png (needs runs/audit_real.html and runs/workstation.html)
"""
import subprocess
import sys
from pathlib import Path

EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
ROOT = Path(__file__).resolve().parents[1]


def shoot(html, png, frag="", w=1600, h=950, scale=2, budget=9000):
    exe = next(p for p in EDGE if Path(p).exists())
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    url = Path(html).resolve().as_uri() + frag
    cmd = [exe, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
           f"--force-device-scale-factor={scale}", f"--window-size={w},{h}", f"--virtual-time-budget={budget}", f"--screenshot={Path(png).resolve()}", url]
    subprocess.run(cmd, capture_output=True, timeout=240)
    print("wrote", png)


if __name__ == "__main__":
    runs = ROOT / "runs"
    docs = ROOT / "docs" / "img"
    # documentation set
    for name, frag in (("audit_overview", "#ws=audit&tab=Overview&sel=3"), ("audit_stain", "#ws=audit&tab=Stain&sel=3&dock=560"), ("audit_quality", "#ws=audit&tab=Quality&sel=491&show=exclude&dock=520"),
                       ("audit_texture", "#ws=audit&tab=Texture&sel=295&dock=560"), ("audit_separability", "#ws=audit&tab=Separability&dock=560"), ("audit_stability", "#ws=audit&tab=Stability&dock=560"),
                       ("audit_interpolation", "#ws=audit&tab=Interpolation&dock=560"), ("audit_light", "#ws=audit&tab=Stain&sel=3&theme=light&dock=520")):
        shoot(runs / "audit_real.html", docs / f"{name}.png", frag, scale=1)
    for i in range(8):
        shoot(runs / "audit_real.html", docs / f"tour_{i + 1}.png", f"#ws=audit&tab=Overview&sel=3&tour={i}", scale=1)
    shoot(runs / "workstation.html", docs / "reconstruction.png", "#tab=Summary", scale=1)
