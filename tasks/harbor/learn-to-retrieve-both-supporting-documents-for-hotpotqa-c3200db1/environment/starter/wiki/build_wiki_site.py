#!/usr/bin/env python3
"""Generate wiki/index.html — a single self-contained page that renders the whole wiki.

    python3 wiki/build_wiki_site.py        # regenerates wiki/index.html

Open wiki/index.html directly in a browser (file:// works — all markdown is embedded, no
server or network needed). Left sidebar mirrors the wiki's section structure; pages render
client-side with a small markdown converter that covers this wiki's subset (headings, fenced
code, tables, nested lists, blockquotes, links, emphasis). Relative .md links become in-page
routes; other relative links resolve against the repo as usual. Re-run after editing wiki
pages — the file is a snapshot of wiki/*.md at generation time (stamp in the footer).
"""
import datetime
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
WIKI = HERE if HERE.name == "wiki" else HERE.parents[1] / "wiki"

SECTION_ORDER = ["", "infrastructure", "architecture", "training", "data",
                 "evaluation", "experiments", "implementations"]
SECTION_TITLES = {"": "Overview", "infrastructure": "Infrastructure",
                  "architecture": "Architecture", "training": "Training", "data": "Data",
                  "evaluation": "Evaluation", "experiments": "Experiments",
                  "implementations": "Implementations"}


def page_title(text, fallback):
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def collect():
    pages, nav = {}, []
    by_section = {}
    for p in sorted(WIKI.rglob("*.md")):
        rel = str(p.relative_to(WIKI))
        pages[rel] = p.read_text()
        section = rel.split("/")[0] if "/" in rel else ""
        by_section.setdefault(section, []).append(rel)
    for sec in SECTION_ORDER:
        items = by_section.pop(sec, [])
        if not items:
            continue
        items.sort(key=lambda r: (not r.endswith("README.md"), r))
        nav.append({"section": SECTION_TITLES.get(sec, sec), "items": [
            {"path": r, "title": page_title(pages[r], pathlib.Path(r).stem)} for r in items]})
    for sec, items in sorted(by_section.items()):   # any future sections
        nav.append({"section": sec, "items": [
            {"path": r, "title": page_title(pages[r], pathlib.Path(r).stem)} for r in items]})
    return pages, nav


TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory Layers — Wiki</title>
<style>
:root{
  --bg:#FAFAF8; --ink:#1A1D21; --ink2:#5A6169; --ink3:#8A9096;
  --accent:#0F6E6B; --accent-soft:#0F6E6B22; --code-bg:#F0F0EC; --border:#E3E3DD;
  --side-bg:#F3F3EF;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#15181B; --ink:#E8E6E0; --ink2:#9AA0A6; --ink3:#6E747A;
  --accent:#4FB8B3; --accent-soft:#4FB8B333; --code-bg:#1E2226; --border:#2A2E33;
  --side-bg:#191C20;
}}
:root[data-theme="dark"]{
  --bg:#15181B; --ink:#E8E6E0; --ink2:#9AA0A6; --ink3:#6E747A;
  --accent:#4FB8B3; --accent-soft:#4FB8B333; --code-bg:#1E2226; --border:#2A2E33;
  --side-bg:#191C20;
}
:root[data-theme="light"]{
  --bg:#FAFAF8; --ink:#1A1D21; --ink2:#5A6169; --ink3:#8A9096;
  --accent:#0F6E6B; --accent-soft:#0F6E6B22; --code-bg:#F0F0EC; --border:#E3E3DD;
  --side-bg:#F3F3EF;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink)}
body{font-family:Charter,Georgia,'Times New Roman',serif;font-size:16px;line-height:1.65}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.layout{display:flex;min-height:100vh}
nav{width:270px;flex:0 0 270px;background:var(--side-bg);border-right:1px solid var(--border);
  padding:18px 0 40px;position:sticky;top:0;height:100vh;overflow-y:auto;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
nav .brand{font-size:15px;font-weight:700;padding:4px 20px 14px;letter-spacing:.2px}
nav .brand span{color:var(--accent)}
nav .sec{font-size:10.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink3);padding:16px 20px 6px}
nav a.pg{display:block;padding:4px 20px 4px 28px;font-size:13.5px;color:var(--ink2);
  border-left:2px solid transparent;line-height:1.4}
nav a.pg:hover{color:var(--ink);text-decoration:none;background:var(--accent-soft)}
nav a.pg.on{color:var(--accent);border-left-color:var(--accent);font-weight:600}
main{flex:1;min-width:0;padding:36px 48px 90px}
.crumb{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  font-size:12px;color:var(--ink3);margin-bottom:18px;letter-spacing:.04em}
article{max-width:76ch}
article h1,article h2,article h3,article h4{font-family:-apple-system,BlinkMacSystemFont,
  'Segoe UI',system-ui,sans-serif;line-height:1.25;text-wrap:balance}
article h1{font-size:28px;margin:0 0 18px;border-bottom:1px solid var(--border);padding-bottom:12px}
article h2{font-size:21px;margin:34px 0 10px}
article h3{font-size:17px;margin:26px 0 8px}
article h4{font-size:15px;margin:20px 0 6px}
article p{margin:10px 0}
article ul,article ol{margin:8px 0;padding-left:26px}
article li{margin:3px 0}
article blockquote{margin:14px 0;padding:10px 16px;border-left:3px solid var(--accent);
  background:var(--accent-soft);border-radius:0 6px 6px 0;color:var(--ink)}
article blockquote p{margin:6px 0}
article code{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;font-size:.86em;
  background:var(--code-bg);padding:1.5px 5px;border-radius:4px}
article pre{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;
  padding:14px 16px;overflow-x:auto;margin:14px 0}
article pre code{background:none;padding:0;font-size:12.8px;line-height:1.55}
.tablewrap{overflow-x:auto;margin:14px 0;border:1px solid var(--border);border-radius:8px}
article table{border-collapse:collapse;width:100%;font-size:14px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
article th{text-align:left;font-size:12px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink2);background:var(--side-bg)}
article th,article td{padding:8px 12px;border-bottom:1px solid var(--border);vertical-align:top}
article tr:last-child td{border-bottom:none}
article hr{border:none;border-top:1px solid var(--border);margin:26px 0}
article img{max-width:100%}
footer{font-family:-apple-system,system-ui,sans-serif;font-size:11.5px;color:var(--ink3);
  margin-top:60px;border-top:1px solid var(--border);padding-top:12px;max-width:76ch}
.menu-btn{display:none}
@media (max-width:860px){
  nav{position:fixed;left:0;top:0;z-index:20;transform:translateX(-100%);transition:transform .18s}
  nav.open{transform:none}
  .menu-btn{display:block;position:fixed;top:10px;right:12px;z-index:30;
    font:600 13px -apple-system,system-ui,sans-serif;background:var(--side-bg);color:var(--ink);
    border:1px solid var(--border);border-radius:8px;padding:7px 12px;cursor:pointer}
  main{padding:52px 20px 80px}
}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style></head>
<body>
<button class="menu-btn" id="menuBtn" aria-label="Toggle navigation">Menu</button>
<div class="layout">
<nav id="nav"><div class="brand">memory-layers <span>/ wiki</span></div>__NAV__</nav>
<main><div class="crumb" id="crumb"></div><article id="content"></article>
<footer>Generated __STAMP__ · <code>python3 wiki/build_wiki_site.py</code> to refresh · snapshot of wiki/*.md</footer>
</main></div>
<script id="pages" type="application/json">__PAGES__</script>
<script>
const PAGES = JSON.parse(document.getElementById('pages').textContent);
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function inline(s){
  s = s.replace(/`([^`]+)`/g, (m,c)=>'<code>'+c+'</code>');
  s = s.replace(/\\[([^\\]]+)\\]\\(([^)\\s]+)\\)/g, (m,txt,href)=>{
    if(/^https?:/.test(href)) return '<a href="'+href+'" target="_blank" rel="noopener">'+txt+'</a>';
    const md = href.match(/^([^#]*\\.md)(#.*)?$/);
    if(md) return '<a href="#'+md[1]+'" data-rel="1">'+txt+'</a>';
    return '<a href="'+href+'">'+txt+'</a>';
  });
  s = s.replace(/\\*\\*([^*]+)\\*\\*/g,'<strong>$1</strong>');
  s = s.replace(/(^|[\\s(>])\\*([^*\\s][^*]*)\\*/g,'$1<em>$2</em>');
  s = s.replace(/(^|[\\s(>])_([^_\\s][^_]*)_(?=[\\s).,;:]|$)/g,'$1<em>$2</em>');
  s = s.replace(/〃/g,'〃');
  return s;
}

function render(md){
  const lines = md.split('\\n');
  let html='', i=0, listStack=[];
  const closeLists = depth=>{ while(listStack.length>depth){ html+='</'+listStack.pop()+'>'; } };
  while(i<lines.length){
    let L=lines[i];
    if(/^```/.test(L)){ closeLists(0); let buf=[]; i++;
      while(i<lines.length && !/^```/.test(lines[i])){ buf.push(lines[i]); i++; }
      html+='<pre><code>'+esc(buf.join('\\n'))+'</code></pre>'; i++; continue; }
    if(/^\\s*$/.test(L)){ closeLists(0); i++; continue; }
    const h=L.match(/^(#{1,4})\\s+(.*)/);
    if(h){ closeLists(0); const n=h[1].length;
      html+='<h'+n+'>'+inline(esc(h[2]))+'</h'+n+'>'; i++; continue; }
    if(/^(---+|\\*\\*\\*+)\\s*$/.test(L)){ closeLists(0); html+='<hr>'; i++; continue; }
    if(/^\\s*>/.test(L)){ closeLists(0); let buf=[];
      while(i<lines.length && /^\\s*>/.test(lines[i])){ buf.push(lines[i].replace(/^\\s*>\\s?/,'')); i++; }
      html+='<blockquote>'+render(buf.join('\\n'))+'</blockquote>'; continue; }
    if(/^\\s*\\|/.test(L) && i+1<lines.length && /^\\s*\\|[\\s:|-]+\\|?\\s*$/.test(lines[i+1])){
      closeLists(0);
      const row = r=>r.trim().replace(/^\\||\\|$/g,'').split('|').map(c=>c.trim());
      const head=row(L); i+=2; let body=[];
      while(i<lines.length && /^\\s*\\|/.test(lines[i])){ body.push(row(lines[i])); i++; }
      html+='<div class="tablewrap"><table><thead><tr>'+head.map(c=>'<th>'+inline(esc(c))+'</th>').join('')+'</tr></thead><tbody>';
      for(const r of body){ html+='<tr>'+r.map(c=>'<td>'+inline(esc(c))+'</td>').join('')+'</tr>'; }
      html+='</tbody></table></div>'; continue; }
    const li=L.match(/^(\\s*)([-*]|\\d+\\.)\\s+(.*)/);
    if(li){ const depth=Math.floor(li[1].length/2)+1, tag=/\\d/.test(li[2])?'ol':'ul';
      while(listStack.length<depth){ html+='<'+tag+'>'; listStack.push(tag); }
      closeLists(depth);
      let item=li[3]; let j=i+1;
      while(j<lines.length && /^\\s{2,}\\S/.test(lines[j]) && !/^\\s*([-*]|\\d+\\.)\\s/.test(lines[j]) && !/^\\s*\\|/.test(lines[j])){
        item+=' '+lines[j].trim(); j++; }
      html+='<li>'+inline(esc(item))+'</li>'; i=j; continue; }
    let buf=[L]; let j=i+1;
    while(j<lines.length && !/^\\s*$/.test(lines[j]) && !/^(#{1,4})\\s|^```|^\\s*>|^\\s*\\||^(---+)\\s*$|^\\s*([-*]|\\d+\\.)\\s/.test(lines[j])){
      buf.push(lines[j]); j++; }
    closeLists(0);
    html+='<p>'+inline(esc(buf.join(' ')))+'</p>'; i=j;
  }
  closeLists(0);
  return html;
}

function resolve(from, rel){
  const parts=(from.split('/').slice(0,-1).join('/')+'/'+rel).split('/');
  const out=[];
  for(const p of parts){ if(p==='..') out.pop(); else if(p!=='.'&&p!=='') out.push(p); }
  return out.join('/');
}

let current='README.md';
function show(path){
  if(!(path in PAGES)){ path='README.md'; }
  current=path;
  document.getElementById('content').innerHTML=render(PAGES[path]);
  document.getElementById('crumb').textContent='wiki / '+path;
  document.querySelectorAll('nav a.pg').forEach(a=>a.classList.toggle('on', a.dataset.path===path));
  document.querySelectorAll('#content a[data-rel]').forEach(a=>{
    const target=resolve(path, a.getAttribute('href').slice(1));
    a.onclick=e=>{ e.preventDefault(); location.hash='#'+target; };
  });
  document.getElementById('nav').classList.remove('open');
  window.scrollTo(0,0);
}
window.addEventListener('hashchange',()=>show(decodeURIComponent(location.hash.slice(1))));
document.getElementById('menuBtn').onclick=()=>document.getElementById('nav').classList.toggle('open');
show(location.hash? decodeURIComponent(location.hash.slice(1)) : 'README.md');
</script>
</body></html>
"""


def main():
    pages, nav = collect()
    nav_html = ""
    for group in nav:
        nav_html += f'<div class="sec">{group["section"]}</div>'
        for item in group["items"]:
            nav_html += (f'<a class="pg" data-path="{item["path"]}" '
                         f'href="#{item["path"]}">{item["title"]}</a>')
    html = (TEMPLATE
            .replace("__NAV__", nav_html)
            .replace("__STAMP__", datetime.date.today().isoformat())
            .replace("__PAGES__", json.dumps(pages).replace("</", "<\\/")))
    out = WIKI / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(pages)} pages)")


if __name__ == "__main__":
    main()
