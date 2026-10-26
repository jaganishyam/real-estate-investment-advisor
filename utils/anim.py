"""
The two animated moments in the app, written as plain HTML/SVG/CSS (plus a
few lines of JS for counting numbers up). No Lottie files or external
assets, so they load instantly and work offline / on Streamlit Cloud.

1. skyline_hero()  - a city at dusk: towers rise, windows switch on one by
                     one, a crane works on the tower that's still going up.
2. verdict_card()  - after you assess a property: a house builds itself,
                     the verdict lands on it, the price counts up to its
                     5-year value and the growth line draws itself.

Both respect prefers-reduced-motion.
"""

import random

import streamlit as st


def show(html: str, height: int):
    """Render one of the animations. st.iframe is the newer API; older Streamlit
    versions only have components.html, so fall back to that."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)

FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;"
         "12..96,800&family=Source+Sans+3:wght@400;600&display=swap');")


# ---------------------------------------------------------------------------
# 1. skyline hero
# ---------------------------------------------------------------------------
def _skyline_svg(seed: int = 11) -> str:
    rnd = random.Random(seed)
    parts, x = [], 0
    crane_x = 820           # the tower still going up sits right of the headline
    crane_done = False
    i = 0
    while x < 1200:
        w = rnd.choice([54, 62, 70, 78, 90])
        is_crane = not crane_done and x >= crane_x
        # low-rise under the text on the left, towers on the right
        h = (120 if is_crane else
             rnd.randint(40, 85) if x < 700 else rnd.randint(95, 200))
        y = 220 - h
        delay = 0.15 + i * 0.07
        shade = rnd.choice(["#262E55", "#2C3561", "#323C6E", "#232A4D"])

        if is_crane:
            crane_done = True
            # half-built: concrete core + scaffold, and the crane on top
            parts.append(f'<g class="rise" style="animation-delay:{delay:.2f}s">'
                         f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#2A3157"/>')
            for sy in range(y + 10, 220, 16):
                parts.append(f'<line x1="{x}" y1="{sy}" x2="{x + w}" y2="{sy}" stroke="#46507F" stroke-width="1"/>')
            for sx in range(x + 8, x + w, 14):
                parts.append(f'<line x1="{sx}" y1="{y}" x2="{sx}" y2="220" stroke="#46507F" stroke-width="1"/>')
            parts.append("</g>")
            cx = x + w // 2
            parts.append(f'''
            <g class="crane" style="animation-delay:{delay + 0.6:.2f}s">
              <rect x="{cx - 3}" y="{y - 92}" width="6" height="92" fill="#F2B544"/>
              <g class="jib" style="transform-origin:{cx}px {y - 88}px">
                <rect x="{cx - 34}" y="{y - 92}" width="130" height="5" fill="#F2B544"/>
                <rect x="{cx - 34}" y="{y - 88}" width="14" height="10" fill="#C98F2E"/>
                <g class="hook">
                  <line x1="{cx + 80}" y1="{y - 87}" x2="{cx + 80}" y2="{y - 40}" stroke="#C9CDE0" stroke-width="1.2"/>
                  <rect x="{cx + 70}" y="{y - 40}" width="20" height="9" fill="#8C93B8"/>
                </g>
              </g>
              <circle cx="{cx}" cy="{y - 96}" r="2.5" fill="#E34948" class="beacon"/>
            </g>''')
        else:
            parts.append(f'<g class="rise" style="animation-delay:{delay:.2f}s">'
                         f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{shade}"/>')
            # windows - most switch on as the evening sets in, a few flicker
            for wy in range(y + 10, 212, 14):
                for wx in range(x + 7, x + w - 8, 12):
                    r = rnd.random()
                    if r < 0.38:
                        cls = "win flick" if r < 0.04 else "win"
                        d = delay + 0.8 + rnd.random() * 3.2
                        parts.append(f'<rect class="{cls}" x="{wx}" y="{wy}" width="6" height="7" '
                                     f'style="animation-delay:{d:.2f}s"/>')
                    else:
                        parts.append(f'<rect x="{wx}" y="{wy}" width="6" height="7" fill="#1E2547"/>')
            if rnd.random() < 0.35:   # rooftop water tank / antenna
                parts.append(f'<rect x="{x + w // 2 - 1}" y="{y - 16}" width="2" height="16" fill="#3A4577"/>')
            parts.append("</g>")
        x += w + rnd.choice([0, 2, 4])
        i += 1
    return "\n".join(parts)


def skyline_hero(headline: str, sub: str, chips: list[tuple[str, str]], height: int = 400) -> tuple[str, int]:
    chips_html = "".join(f'<div class="chip"><b>{v}</b><span>{k}</span></div>' for v, k in chips)
    stars = "".join(
        f'<circle cx="{random.Random(s).randint(10, 1190)}" cy="{random.Random(s * 7).randint(8, 120)}" '
        f'r="{random.Random(s * 3).choice([0.8, 1, 1.3])}" class="star" style="animation-delay:{(s % 9) * 0.4:.1f}s"/>'
        for s in range(1, 46))
    html = f"""
<!doctype html><html><head><meta charset="utf-8"><style>
{FONTS}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: transparent; font-family: 'Source Sans 3', sans-serif; }}
.hero {{ position: relative; height: {height - 8}px; border-radius: 16px; overflow: hidden;
        background: linear-gradient(180deg, #121733 0%, #1B2240 42%, #352F5E 74%, #6B4A68 100%); }}
.copy {{ position: absolute; left: 36px; top: 30px; right: 36px; z-index: 3; max-width: 560px; }}
h1 {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; color: #fff;
      font-size: clamp(26px, 3.4vw, 40px); line-height: 1.08; letter-spacing: -0.02em;
      opacity: 0; animation: fadeUp .9s .2s ease-out forwards; }}
p {{ color: #C3C7DD; font-size: 17px; line-height: 1.5; margin-top: 12px; max-width: 54ch;
     opacity: 0; animation: fadeUp .9s .45s ease-out forwards; }}
.chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px;
          opacity: 0; animation: fadeUp .9s .7s ease-out forwards; }}
.chip {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
         border-radius: 10px; padding: 7px 12px; color: #D9DCEB; font-size: 13px; }}
.chip b {{ color: #F2B544; font-family: 'Bricolage Grotesque', sans-serif; font-size: 17px; margin-right: 6px; }}
svg.sky {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
svg.city {{ position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 62%; }}
.moon {{ fill: #F4E7C5; filter: drop-shadow(0 0 18px rgba(244,231,197,.55)); }}
.star {{ fill: #fff; opacity: .5; animation: twinkle 3.6s ease-in-out infinite; }}
.rise {{ transform-box: fill-box; transform-origin: bottom; transform: scaleY(0);
         animation: rise 1.1s cubic-bezier(.2,.8,.2,1) forwards; }}
.win {{ fill: #1E2547; animation: lightOn .6s ease-out forwards; }}
.win.flick {{ animation: lightOn .6s ease-out forwards, flicker 5s 6s steps(1) infinite; }}
.crane {{ opacity: 0; animation: fadeIn .8s forwards; }}
.jib {{ animation: swing 7s ease-in-out infinite alternate; }}
.hook {{ animation: hoist 4.5s ease-in-out infinite alternate; }}
.beacon {{ animation: blink 1.4s steps(1) infinite; }}
.road {{ fill: #141936; }}
.car {{ animation: drive 11s linear infinite; }}
.car2 {{ animation: drive2 15s linear 3s infinite; }}
@keyframes rise {{ to {{ transform: scaleY(1); }} }}
@keyframes lightOn {{ to {{ fill: #F2B544; }} }}
@keyframes flicker {{ 0%,100% {{ fill: #F2B544; }} 48% {{ fill: #1E2547; }} 52% {{ fill: #F2B544; }} }}
@keyframes twinkle {{ 50% {{ opacity: .1; }} }}
@keyframes swing {{ from {{ transform: rotate(-4deg); }} to {{ transform: rotate(5deg); }} }}
@keyframes hoist {{ from {{ transform: translateY(0); }} to {{ transform: translateY(-26px); }} }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes fadeIn {{ to {{ opacity: 1; }} }}
@keyframes drive {{ from {{ transform: translateX(-60px); }} to {{ transform: translateX(1260px); }} }}
@keyframes drive2 {{ from {{ transform: translateX(1260px); }} to {{ transform: translateX(-60px); }} }}
@media (max-width: 640px) {{ .copy {{ left: 18px; right: 18px; top: 20px; }} p {{ font-size: 15px; }} }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: 0s !important; animation-delay: 0s !important;
                            animation-iteration-count: 1 !important; }}
}}
</style></head><body>
<div class="hero" role="img" aria-label="Animated city skyline at dusk">
  <svg class="sky" viewBox="0 0 1200 400" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
    {stars}
    <circle class="moon" cx="1080" cy="70" r="26"/>
    <circle cx="1091" cy="62" r="24" fill="#1A2042" opacity=".9"/>
  </svg>
  <div class="copy">
    <h1>{headline}</h1>
    <p>{sub}</p>
    <div class="chips">{chips_html}</div>
  </div>
  <svg class="city" viewBox="0 0 1200 240" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
    {_skyline_svg()}
    <rect class="road" x="0" y="220" width="1200" height="20"/>
    <rect x="0" y="229" width="1200" height="1.5" fill="#2E3765"/>
    <g class="car"><rect x="0" y="222" width="22" height="7" rx="2" fill="#6250d6"/>
       <circle cx="23" cy="225" r="1.6" fill="#FFF4C8"/></g>
    <g class="car2"><rect x="0" y="231" width="20" height="6" rx="2" fill="#2a78d6"/>
       <circle cx="-1" cy="234" r="1.4" fill="#FFF4C8"/></g>
  </svg>
</div>
</body></html>"""
    return html, height


# ---------------------------------------------------------------------------
# 2. verdict card
# ---------------------------------------------------------------------------
def verdict_card(*, good: bool, prob: float, price_now: float, price_5y: float,
                 low_5y: float, high_5y: float, cagr: float, start_year: int,
                 city: str, height: int = 470) -> tuple[str, int]:
    accent = "#1F8A70" if good else "#C2413B"
    soft = "#E3F3EE" if good else "#F8E6E4"
    verdict = "Good investment" if good else "Think twice"
    reason = ("Strong expected growth for its price in " + city if good
              else "Growth or pricing doesn't clear the bar in " + city)
    icon = ('<path d="M6 12.5l4 4 8-9" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" '
            'stroke-linejoin="round"/>' if good else
            '<path d="M12 6v8M12 17.5v.5" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/>')

    # year-by-year path at the implied growth rate
    years = list(range(6))
    vals = [price_now * (1 + cagr) ** t for t in years]
    lows = [price_now + (low_5y - price_now) * t / 5 for t in years]
    highs = [price_now + (high_5y - price_now) * t / 5 for t in years]
    lo_v, hi_v = min(lows + vals) * 0.96, max(highs + vals) * 1.03
    W, H, PADL, PADB = 420, 170, 8, 24

    def px(t):
        return PADL + t * (W - PADL - 40) / 5

    def py(v):
        return 8 + (H - PADB - 8) * (1 - (v - lo_v) / (hi_v - lo_v or 1))

    line = " ".join(f"{'M' if t == 0 else 'L'}{px(t):.1f},{py(v):.1f}" for t, v in zip(years, vals))
    band = (" ".join(f"{'M' if t == 0 else 'L'}{px(t):.1f},{py(v):.1f}" for t, v in zip(years, highs))
            + " " + " ".join(f"L{px(t):.1f},{py(v):.1f}" for t, v in zip(years[::-1], lows[::-1])) + " Z")
    dots = "".join(f'<circle class="dot" cx="{px(t):.1f}" cy="{py(v):.1f}" r="4" '
                   f'style="animation-delay:{2.2 + t * 0.18:.2f}s"/>' for t, v in zip(years, vals))
    xlabels = "".join(f'<text x="{px(t):.1f}" y="{H - 4}" text-anchor="middle">'
                      f'{"Today" if t == 0 else start_year + t}</text>' for t in years)
    ring = 2 * 3.14159 * 34

    html = f"""
<!doctype html><html><head><meta charset="utf-8"><style>
{FONTS}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: transparent; font-family: 'Source Sans 3', sans-serif; color: #1A1F36; }}
.card {{ display: grid; grid-template-columns: minmax(240px, 1fr) 1.35fr; gap: 0; height: {height - 10}px;
         background: #fff; border: 1px solid #E3E6EF; border-radius: 16px; overflow: hidden; }}
.stage {{ position: relative; background: linear-gradient(180deg, #EEF1FA 0%, #F7F8FC 70%, #E9ECF5 100%);
          display: flex; align-items: center; justify-content: center; }}
.stage svg {{ width: 88%; max-width: 330px; }}
.info {{ padding: 26px 28px; display: flex; flex-direction: column; gap: 16px; }}
.k {{ color: #5B6382; font-size: 14px; }}
.big {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 40px; line-height: 1; letter-spacing: -0.02em; }}
.row {{ display: flex; gap: 28px; align-items: flex-end; flex-wrap: wrap; }}
.now {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 22px; color: #5B6382; }}
.cagr {{ display: inline-block; margin-top: 6px; background: {soft}; color: {accent}; font-weight: 600;
         padding: 3px 10px; border-radius: 999px; font-size: 14px; }}
.chart text {{ fill: #5B6382; font-size: 11px; font-family: 'Source Sans 3', sans-serif; }}
.band {{ fill: {accent}; opacity: 0; animation: bandIn .8s 2.1s forwards; }}
.line {{ fill: none; stroke: {accent}; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round;
         stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 1.6s 1.9s ease-out forwards; }}
.dot {{ fill: #fff; stroke: {accent}; stroke-width: 2; opacity: 0; animation: pop .35s ease-out forwards; }}
.foot {{ display: flex; gap: 18px; align-items: center; margin-top: auto; }}
.ring-bg {{ fill: none; stroke: #EDEFF6; stroke-width: 8; }}
.ring {{ fill: none; stroke: {accent}; stroke-width: 8; stroke-linecap: round; transform: rotate(-90deg);
         transform-origin: 40px 40px; stroke-dasharray: {ring:.1f}; stroke-dashoffset: {ring:.1f};
         animation: fill 1.4s 2.4s cubic-bezier(.2,.8,.2,1) forwards; }}
.pct {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 17px; fill: #1A1F36; }}

/* the house */
.ground {{ stroke: #B9BFD6; stroke-width: 2; }}
.found {{ fill: #8C93B8; transform-box: fill-box; transform-origin: bottom; transform: scaleY(0); animation: grow .35s .1s ease-out forwards; }}
.walls {{ fill: #FFFFFF; stroke: #1A1F36; stroke-width: 2.5; transform-box: fill-box; transform-origin: bottom;
          transform: scaleY(0); animation: grow .6s .4s cubic-bezier(.2,.8,.2,1) forwards; }}
.roof {{ fill: #1B2240; opacity: 0; transform: translateY(-50px); animation: drop .55s 1s cubic-bezier(.3,1.4,.5,1) forwards; }}
.chim {{ fill: #1B2240; opacity: 0; animation: fade .3s 1.35s forwards; }}
.door {{ fill: {accent}; transform-box: fill-box; transform-origin: bottom; transform: scaleY(0); animation: grow .35s 1.45s forwards; }}
.wnd {{ fill: #DDE1EE; stroke: #1A1F36; stroke-width: 2; opacity: 0; animation: fade .3s 1.55s forwards, glow .5s 1.9s forwards; }}
.tree {{ transform-box: fill-box; transform-origin: bottom; transform: scale(0); animation: grow2 .6s 1.7s cubic-bezier(.3,1.5,.5,1) forwards; }}
.smoke {{ fill: #C9CDE0; opacity: 0; animation: smoke 3s 2s ease-out infinite; }}
.stamp {{ opacity: 0; transform-box: fill-box; transform-origin: center; animation: stamp .5s 2.1s cubic-bezier(.3,1.6,.5,1) forwards; }}
.stamp text {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 15px; fill: #fff; }}

@keyframes grow {{ to {{ transform: scaleY(1); }} }}
@keyframes grow2 {{ to {{ transform: scale(1); }} }}
@keyframes drop {{ to {{ opacity: 1; transform: none; }} }}
@keyframes fade {{ to {{ opacity: 1; }} }}
@keyframes bandIn {{ to {{ opacity: .14; }} }}
@keyframes glow {{ to {{ fill: #F2B544; }} }}
@keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
@keyframes pop {{ from {{ opacity: 0; r: 0; }} to {{ opacity: 1; r: 4; }} }}
@keyframes fill {{ to {{ stroke-dashoffset: {ring * (1 - prob):.1f}; }} }}
@keyframes smoke {{ 0% {{ opacity: 0; transform: translate(0,0) scale(.6); }} 30% {{ opacity: .8; }}
                    100% {{ opacity: 0; transform: translate(10px,-30px) scale(1.4); }} }}
@keyframes stamp {{ from {{ opacity: 0; transform: scale(2.2) rotate(-8deg); }} to {{ opacity: 1; transform: scale(1) rotate(-8deg); }} }}
@media (max-width: 640px) {{ .card {{ grid-template-columns: 1fr; height: auto; }} .stage {{ height: 220px; }} }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: 0s !important; animation-delay: 0s !important;
                            animation-iteration-count: 1 !important; }}
}}
</style></head><body>
<div class="card">
  <div class="stage">
    <svg viewBox="0 0 300 250" role="img" aria-label="{verdict}">
      <line class="ground" x1="10" y1="220" x2="290" y2="220"/>
      <rect class="found" x="62" y="210" width="176" height="10" rx="2"/>
      <rect class="chim" x="190" y="78" width="16" height="34"/>
      <g class="smoke"><circle cx="198" cy="70" r="6"/><circle cx="204" cy="62" r="5"/></g>
      <rect class="walls" x="72" y="128" width="156" height="82"/>
      <polygon class="roof" points="56,132 150,62 244,132"/>
      <rect class="door" x="134" y="168" width="32" height="42" rx="3"/>
      <rect class="wnd" x="88" y="146" width="32" height="26" rx="2"/>
      <rect class="wnd" x="180" y="146" width="32" height="26" rx="2"/>
      <g class="tree"><rect x="262" y="196" width="5" height="24" fill="#6B5A48"/>
         <circle cx="264.5" cy="186" r="16" fill="{'#1F8A70' if good else '#8C93B8'}"/></g>
      <g class="stamp">
        <rect x="52" y="18" width="196" height="38" rx="19" fill="{accent}"/>
        <circle cx="74" cy="37" r="11" fill="rgba(255,255,255,.18)"/>
        <g transform="translate(62,25)">{icon}</g>
        <text x="160" y="42" text-anchor="middle">{verdict}</text>
      </g>
    </svg>
  </div>
  <div class="info">
    <div>
      <div class="k">Estimated value in {start_year + 5}</div>
      <div class="row">
        <div class="big" id="future" data-to="{price_5y:.4f}" data-from="{price_now:.4f}">-</div>
        <div><div class="k">Today</div><div class="now" id="now" data-v="{price_now:.4f}">-</div></div>
      </div>
      <span class="cagr">≈ {cagr * 100:.1f}% a year · likely range <span id="lo" data-v="{low_5y:.4f}"></span> – <span id="hi" data-v="{high_5y:.4f}"></span></span>
    </div>
    <svg class="chart" viewBox="0 0 {W} {H}" width="100%" height="{H}">
      <path class="band" d="{band}"/>
      <path class="line" d="{line}"/>
      {dots}
      {xlabels}
    </svg>
    <div class="foot">
      <svg width="80" height="80" viewBox="0 0 80 80" aria-hidden="true">
        <circle class="ring-bg" cx="40" cy="40" r="34"/>
        <circle class="ring" cx="40" cy="40" r="34"/>
        <text class="pct" x="40" y="46" text-anchor="middle" id="pct" data-v="{prob:.4f}">0%</text>
      </svg>
      <div><div style="font-weight:600">Chance this is a good investment</div>
           <div class="k">{reason}.</div></div>
    </div>
  </div>
</div>
<script>
const fmt = v => v >= 100 ? '₹' + (v/100).toFixed(2) + ' Cr' : '₹' + v.toFixed(1) + ' L';
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function countUp(el, from, to, delay, dur, f) {{
  if (reduce) {{ el.textContent = f(to); return; }}
  el.textContent = f(from);
  setTimeout(() => {{
    const t0 = performance.now();
    const step = now => {{
      const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = f(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(step);
    }};
    requestAnimationFrame(step);
  }}, delay);
}}
const fut = document.getElementById('future');
countUp(fut, +fut.dataset.from, +fut.dataset.to, 1900, 1600, fmt);
document.getElementById('now').textContent = fmt(+document.getElementById('now').dataset.v);
document.getElementById('lo').textContent = fmt(+document.getElementById('lo').dataset.v);
document.getElementById('hi').textContent = fmt(+document.getElementById('hi').dataset.v);
const pct = document.getElementById('pct');
countUp(pct, 0, +pct.dataset.v * 100, 2400, 1400, v => Math.round(v) + '%');
</script>
</body></html>"""
    return html, height
