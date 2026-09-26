"""Extract the supplied visual primitives without copying its second simulator."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "prototype/base_siege_game.html").read_text()
css = re.search(r"<style>(.*?)</style>", source, re.S).group(1)
glyphs = source[source.index("function shade("):source.index("function draw(now,prog)")]
target = ROOT / "ui/frontend"
target.mkdir(parents=True, exist_ok=True)
(target / "theme.css").write_text(css + "\ncanvas{width:100%;height:auto}.boardwrap{flex:1;min-width:300px;max-width:572px} .wrap{padding:8px} .error{color:#a93232} button:focus-visible{outline:3px solid #3179C6} .toolbar{margin:0 0 14px} @media(max-width:640px){.side{min-width:100%}.hud{grid-template-columns:repeat(2,1fr)}}\n")
(target / "glyphs.js").write_text("// Visual primitives retained from Vaibhav's supplied HTML.\n" + glyphs)
print("Extracted theme.css and glyphs.js from the archived visual reference")
