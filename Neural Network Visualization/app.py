import cv2
import numpy as np
import joblib
import os
import mediapipe as mp
import random

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)

# ── Model ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(SCRIPT_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(SCRIPT_DIR, 'scaler.pkl'))
W0, W1, W2 = model.coefs_
b0, b1, b2 = model.intercepts_

INPUT_IDX = np.linspace(0, 783, 16, dtype=int)
W0_sub    = W0[INPUT_IDX]          # shape (16, 32)
W0_MAX    = np.abs(W0_sub).max() + 1e-8
W1_MAX    = np.abs(W1).max()      + 1e-8
W2_MAX    = np.abs(W2).max()      + 1e-8

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def forward(flat):
    x  = scaler.transform((flat * 255.0).reshape(1, -1))[0]
    a1 = np.maximum(0, x @ W0 + b0)
    a2 = np.maximum(0, a1 @ W1 + b1)
    a3 = softmax(a2 @ W2 + b2)
    return x, a1 / (a1.max() + 1e-8), a2 / (a2.max() + 1e-8), a3

# ── Palette (all BGR) ──────────────────────────────────────────────────────────
BG_PANEL   = (22, 14, 10)       # very dark navy for panel overlays
PRIMARY    = (255, 145, 85)     # electric blue  RGB(85,145,255)
SECONDARY  = (55, 105, 255)     # warm coral     RGB(255,105,55)  — winner only
INACTIVE_N = (65, 48, 32)       # dim dark neuron when not firing
TEXT_SOFT  = (210, 205, 220)    # near-white labels
TEXT_DIM   = (125, 115, 135)    # dimmer labels
GRID_COL   = (0, 200, 255)      # cyan grid border

LAYER_SIZES = [16, 32, 16, 10]
LAYER_NAMES = ['INPUT', 'HIDDEN 1', 'HIDDEN 2', 'OUTPUT']
BASE_NR     = [11, 6, 9, 14]    # neuron radii at 720p

# ── Layout (recomputed each frame from current window size) ────────────────────
def compute_layout(w, h):
    scale    = h / 720.0
    margin_y = max(28, int(h * 0.065))
    cell     = max(6, int((h - 2 * margin_y) / 28))
    gx       = max(8, int(w * 0.015))
    gy       = margin_y
    gdisp    = 28 * cell

    pbox_w   = max(90, int(w * 0.12))
    pbox_x   = w - pbox_w - max(8, int(w * 0.01))
    pbox_y   = gy
    pbox_h   = gdisp

    nn_x1    = gx + gdisp + max(22, int(w * 0.028))
    nn_x2    = pbox_x - max(10, int(w * 0.012))
    nn_y1    = gy
    nn_y2    = gy + gdisp

    return dict(cell=cell, gx=gx, gy=gy, gdisp=gdisp,
                nn_x1=nn_x1, nn_x2=nn_x2, nn_y1=nn_y1, nn_y2=nn_y2,
                pbox_x=pbox_x, pbox_y=pbox_y, pbox_w=pbox_w, pbox_h=pbox_h,
                w=w, h=h, scale=scale)

def neuron_positions(L):
    nn_h = L['nn_y2'] - L['nn_y1']
    xs   = np.linspace(L['nn_x1'] + 32, L['nn_x2'] - 12, 4, dtype=int)
    pos  = []
    for n, cx in zip(LAYER_SIZES, xs):
        sp = nn_h / (n + 1)
        pos.append([(int(cx), int(L['nn_y1'] + sp * (j + 1))) for j in range(n)])
    return pos

def nr(layer_idx, L):
    return max(3, int(BASE_NR[layer_idx] * L['scale']))

# ── Color helpers ──────────────────────────────────────────────────────────────
def primary_bgr(v):
    v = float(np.clip(v, 0, 1))
    return (min(255, int(PRIMARY[0] * (0.28 + 0.72 * v))),
            min(255, int(PRIMARY[1] * (0.28 + 0.72 * v))),
            min(255, int(PRIMARY[2] * (0.28 + 0.72 * v))))

def secondary_bgr(v):
    v = float(np.clip(v, 0, 1))
    return tuple(min(255, int(c * (0.38 + 0.62 * v))) for c in SECONDARY)

def glow_circle(img, cx, cy, r, color):
    gr    = r + max(3, r // 2)
    dim1  = tuple(max(0, c // 6) for c in color)
    dim2  = tuple(max(0, c // 3) for c in color)
    brigh = tuple(min(255, int(c * 1.3 + 40)) for c in color)
    cv2.circle(img, (cx, cy), gr,          dim1,  -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r + 3,       dim2,  -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r,           color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, r//2), brigh, -1, cv2.LINE_AA)

# ── Canvas ─────────────────────────────────────────────────────────────────────
canvas  = np.zeros((28, 28), dtype=np.float32)
prev_px = set()

# ── Pulses: grid cell → INPUT neuron ──────────────────────────────────────────
pulses   = []
MAX_PLS  = 14
PULSE_LF = 22

def spawn_pulse(gcol, grow):
    ni = min(int(grow / 28 * 16), 15)
    pulses.append({'gcol': gcol, 'grow': grow, 'ni': ni, 't': 0.0})

def update_pulses(frame, npos, L):
    alive = []
    for p in pulses:
        sx = L['gx'] + p['gcol'] * L['cell'] + L['cell'] // 2
        sy = L['gy'] + p['grow'] * L['cell'] + L['cell'] // 2
        ex, ey = npos[0][p['ni']]
        t   = p['t']
        px  = int(sx + (ex - sx) * t)
        py  = int(sy + (ey - sy) * t)
        fad = 1.0 - t
        col = (int(fad * PRIMARY[0] * 0.85), int(fad * PRIMARY[1] * 0.85), int(fad * 200))
        r_  = max(2, int(5 * L['scale']))
        cv2.circle(frame, (px, py), r_, col, -1, cv2.LINE_AA)
        p['t'] += 1.0 / PULSE_LF
        if p['t'] < 1.0:
            alive.append(p)
    pulses[:] = alive

# ── Connection particles: INPUT → HIDDEN 1 ────────────────────────────────────
conn_particles  = []
MAX_CONN_PARTS  = 80
PART_THRESH     = 0.10
PART_RATE       = 0.20

def spawn_conn_particles(a_in):
    if len(conn_particles) >= MAX_CONN_PARTS:
        return
    for i in range(16):
        if a_in[i] < 0.05:
            continue
        for j in range(32):
            wn       = abs(W0_sub[i, j]) / W0_MAX
            strength = float(np.clip(a_in[i] * wn * 2.5, 0, 1))
            if strength < PART_THRESH:
                continue
            if random.random() > PART_RATE * strength:
                continue
            if len(conn_particles) >= MAX_CONN_PARTS:
                return
            conn_particles.append({
                'i': i, 'j': j, 't': 0.0,
                'speed': 0.015 + strength * 0.045,
                'strength': strength
            })

def update_conn_particles(frame, npos, L):
    alive = []
    for p in conn_particles:
        sx, sy = npos[0][p['i']]
        ex, ey = npos[1][p['j']]
        t  = p['t']
        px = int(sx + (ex - sx) * t)
        py = int(sy + (ey - sy) * t)
        v  = p['strength']
        col = primary_bgr(0.5 + 0.5 * v)
        r_  = max(2, int((2 + v * 2) * L['scale']))
        cv2.circle(frame, (px, py), r_,     col,                        -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r_ + 2, tuple(c // 5 for c in col), 1,  cv2.LINE_AA)
        p['t'] += p['speed']
        if p['t'] < 1.0:
            alive.append(p)
    conn_particles[:] = alive

# ── Draw: grid ─────────────────────────────────────────────────────────────────
def draw_grid(frame, L):
    gx, gy, cell, gdisp = L['gx'], L['gy'], L['cell'], L['gdisp']
    ov = frame.copy()
    cv2.rectangle(ov, (gx-4, gy-4), (gx+gdisp+3, gy+gdisp+3), BG_PANEL, -1)
    cv2.addWeighted(frame, 0.32, ov, 0.68, 0, frame)
    for row in range(28):
        for col in range(28):
            v = canvas[row, col]
            if v > 0.02:
                b_ = int(v * 255)
                x0 = gx + col * cell;  y0 = gy + row * cell
                cv2.rectangle(frame, (x0, y0), (x0+cell-1, y0+cell-1), (b_, b_, b_), -1)
    for i in range(29):
        cv2.line(frame, (gx+i*cell, gy), (gx+i*cell, gy+gdisp), (30, 24, 20), 1)
        cv2.line(frame, (gx, gy+i*cell), (gx+gdisp, gy+i*cell), (30, 24, 20), 1)
    cv2.rectangle(frame, (gx-2, gy-2), (gx+gdisp+1, gy+gdisp+1), GRID_COL, 2)
    cv2.putText(frame, "DRAW DIGIT", (gx, gy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.32, L['h']/2200), TEXT_SOFT, 1, cv2.LINE_AA)

# ── Draw: neural network ────────────────────────────────────────────────────────
def draw_nn(frame, acts, L, npos):
    a_in_full, a1, a2, a3 = acts
    a_in       = a_in_full[INPUT_IDX]
    layer_acts = [a_in, a1, a2, a3]
    Ws         = [W0_sub, W1, W2]
    Wmaxs      = [W0_MAX, W1_MAX, W2_MAX]
    winner     = int(np.argmax(a3))

    ov = frame.copy()
    cv2.rectangle(ov, (L['nn_x1']-18, L['nn_y1']-5),
                      (L['nn_x2']+5,  L['nn_y2']+5), BG_PANEL, -1)
    cv2.addWeighted(frame, 0.26, ov, 0.74, 0, frame)

    # connections — all primary-blue family, skip near-zero
    for l in range(3):
        for i, (sx, sy) in enumerate(npos[l]):
            for j, (dx, dy) in enumerate(npos[l+1]):
                wn       = abs(Ws[l][i, j]) / Wmaxs[l]
                strength = float(np.clip(layer_acts[l][i] * wn * 2.8, 0, 1))
                if strength < 0.06:
                    continue
                thick = max(1, int(strength * 2.5 * L['scale']))
                cv2.line(frame, (sx, sy), (dx, dy), primary_bgr(strength), thick, cv2.LINE_AA)

    # neurons
    for l, (positions, acts_l) in enumerate(zip(npos, layer_acts)):
        r = nr(l, L)
        for i, (cx, cy) in enumerate(positions):
            v = float(np.clip(acts_l[i], 0, 1))
            if l == 3 and i == winner:
                col = secondary_bgr(max(v, 0.4))
            elif v < 0.06:
                col = INACTIVE_N
            else:
                col = primary_bgr(v)
            glow_circle(frame, cx, cy, r, col)
            # digit label overlaid inside each output neuron
            if l == 3:
                fsz = max(0.25, r / 18.0)
                (tw, th), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, fsz, 1)
                lc = TEXT_SOFT if i == winner else TEXT_DIM
                cv2.putText(frame, str(i), (cx - tw//2, cy + th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, fsz, lc, 1, cv2.LINE_AA)

    # layer labels
    for l, name in enumerate(LAYER_NAMES):
        cx  = npos[l][0][0]
        fsz = max(0.28, L['h'] / 2000)
        cv2.putText(frame, name, (cx - 28, L['nn_y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fsz, TEXT_SOFT, 1, cv2.LINE_AA)
        cv2.putText(frame, f'({LAYER_SIZES[l]})', (cx - 16, L['nn_y1'] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, fsz * 0.78, TEXT_DIM, 1, cv2.LINE_AA)

# ── Draw: prediction box ────────────────────────────────────────────────────────
def draw_prediction_box(frame, prediction, confidence, L):
    px, py = L['pbox_x'], L['pbox_y']
    pw, ph = L['pbox_w'], L['pbox_h']

    ov = frame.copy()
    cv2.rectangle(ov, (px-2, py-2), (px+pw+2, py+ph+2), BG_PANEL, -1)
    cv2.addWeighted(frame, 0.26, ov, 0.74, 0, frame)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), SECONDARY, 2)

    lbl_fsz = max(0.28, L['h'] / 2400)
    cv2.putText(frame, "PREDICTION", (px + 6, py + int(ph * 0.09)),
                cv2.FONT_HERSHEY_SIMPLEX, lbl_fsz, TEXT_DIM, 1, cv2.LINE_AA)

    if prediction >= 0:
        d_fsz = max(1.2, ph / 120.0)
        digit_str = str(prediction)
        (tw, th), _ = cv2.getTextSize(digit_str, cv2.FONT_HERSHEY_SIMPLEX, d_fsz, 3)
        dx = px + (pw - tw) // 2
        dy = py + int(ph * 0.50) + th // 2
        # shadow glow
        cv2.putText(frame, digit_str, (dx+2, dy+2),
                    cv2.FONT_HERSHEY_SIMPLEX, d_fsz, tuple(c//4 for c in SECONDARY), 3, cv2.LINE_AA)
        cv2.putText(frame, digit_str, (dx, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, d_fsz, SECONDARY, 3, cv2.LINE_AA)

        conf_str = f"{confidence:.1f}%"
        c_fsz    = max(0.38, L['h'] / 1800)
        (cw, _), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, c_fsz, 1)
        cv2.putText(frame, conf_str, (px + (pw - cw)//2, py + int(ph * 0.74)),
                    cv2.FONT_HERSHEY_SIMPLEX, c_fsz, TEXT_SOFT, 1, cv2.LINE_AA)

        bm = max(6, int(pw * 0.10))
        bt = py + int(ph * 0.82);  bb = py + int(ph * 0.90)
        fw = pw - 2*bm;            fi = int(fw * confidence / 100)
        cv2.rectangle(frame, (px+bm, bt), (px+pw-bm, bb), (40, 30, 20), -1)
        if fi > 0:
            cv2.rectangle(frame, (px+bm, bt), (px+bm+fi, bb), SECONDARY, -1)
        cv2.rectangle(frame, (px+bm, bt), (px+pw-bm, bb), TEXT_DIM, 1)
    else:
        cv2.putText(frame, "---", (px + pw//2 - 22, py + ph//2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.7, L['h']/1000), TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, "draw a digit", (px + max(4, int(pw*0.05)), py + int(ph*0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.28, L['h']/2600), TEXT_DIM, 1, cv2.LINE_AA)

# ── Draw: hand skeleton ────────────────────────────────────────────────────────
def draw_hand(frame, lms_list, w, h):
    for lms in lms_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
        for conn in mp.solutions.hands.HAND_CONNECTIONS:
            a, b = pts[conn[0]], pts[conn[1]]
            cv2.line(frame, a, b, (180, 0, 200), 4, cv2.LINE_AA)
            cv2.line(frame, a, b, (255, 80, 255), 2, cv2.LINE_AA)
        for cx, cy in pts:
            cv2.circle(frame, (cx, cy), 11, (0,  80,  80), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy),  8, (0, 240, 240), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy),  3, (255, 255, 255), -1, cv2.LINE_AA)

# ── Gesture detection (unchanged) ──────────────────────────────────────────────
def is_pinching(lms, w, h):
    tx = lms[4].x * w;  ty = lms[4].y * h
    ix = lms[8].x * w;  iy = lms[8].y * h
    return float(np.hypot(tx - ix, ty - iy)) < w * 0.035

def is_open_palm(lms):
    # Stricter: each fingertip must be clearly (15%) farther from wrist than its PIP joint.
    # A partial curl during drawing won't satisfy this margin.
    MARGIN = 1.15
    wrist = np.array([lms[0].x, lms[0].y])
    for tip_i, pip_i in [(8,6),(12,10),(16,14),(20,18)]:
        tip = np.array([lms[tip_i].x, lms[tip_i].y])
        pip = np.array([lms[pip_i].x, lms[pip_i].y])
        if np.linalg.norm(tip - wrist) <= np.linalg.norm(pip - wrist) * MARGIN:
            return False
    t4 = np.array([lms[4].x, lms[4].y])
    t3 = np.array([lms[3].x, lms[3].y])
    return float(np.linalg.norm(t4 - wrist)) > float(np.linalg.norm(t3 - wrist)) * 1.10

# ── Preprocessing (unchanged) ──────────────────────────────────────────────────
def preprocess_for_model(c):
    if c.max() < 0.05:
        return c.flatten()
    rows = np.any(c > 0.05, axis=1);  cols = np.any(c > 0.05, axis=0)
    if not rows.any() or not cols.any():
        return c.flatten()
    rmin, rmax = np.where(rows)[0][[0,-1]];  cmin, cmax = np.where(cols)[0][[0,-1]]
    digit = c[rmin:rmax+1, cmin:cmax+1]
    h_, w_ = digit.shape;  scale = 20.0 / max(h_, w_)
    new_h  = max(1, int(round(h_ * scale)));  new_w = max(1, int(round(w_ * scale)))
    digit  = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out    = np.zeros((28,28), dtype=np.float32)
    y0_ = (28-new_h)//2;  x0_ = (28-new_w)//2
    out[y0_:y0_+new_h, x0_:x0_+new_w] = digit
    return out.flatten()

BRUSH_R = 1
ERASE_R = 3

def paint_at(gcol, grow, newly_active):
    for ddy in range(-BRUSH_R, BRUSH_R+1):
        for ddx in range(-BRUSH_R, BRUSH_R+1):
            if ddx*ddx + ddy*ddy <= BRUSH_R*BRUSH_R:
                nx, ny = gcol+ddx, grow+ddy
                if 0 <= nx < 28 and 0 <= ny < 28:
                    canvas[ny, nx] = min(1.0, canvas[ny, nx] + 0.55)
                    if canvas[ny, nx] > 0.15 and (nx, ny) not in prev_px:
                        newly_active.add((nx, ny))

def stroke_to(x0, y0, x1, y1, newly_active):
    steps = max(abs(x1-x0), abs(y1-y0), 1)
    for i in range(steps+1):
        t = i / steps
        paint_at(int(round(x0+(x1-x0)*t)), int(round(y0+(y1-y0)*t)), newly_active)

# ── Main loop ──────────────────────────────────────────────────────────────────
WIN_NAME = "Neural Network Visualization"
cap      = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

acts           = (np.zeros(784), np.zeros(32), np.zeros(16), np.zeros(10))
prediction     = -1
confidence     = 0.0
smooth_ix      = None
smooth_iy      = None
last_gcol      = None
last_grow      = None
SMOOTH_A       = 0.4
is_drawing_now = False

# Gesture priority / debounce
PINCH_COOLDOWN_F = 9    # ~0.3 s at 30 fps — after pinch, block erase checks
ERASE_HOLD_F     = 15   # ~0.5 s at 30 fps — palm must be held this long before erasing
pinch_cooldown   = 0    # countdown frames remaining since last pinch
erase_hold       = 0    # consecutive frames palm has been clearly open with no pinch
erase_active     = False

print("Running.")
print("  Pinch thumb+index over grid  -> draw")
print("  Open palm over grid          -> erase along path")
print("  [C] full clear   [Q/ESC] quit")

while True:
    ret, raw = cap.read()
    if not ret:
        break
    raw = cv2.flip(raw, 1)

    # run MediaPipe on raw camera frame (fixed resolution, faster)
    rgb    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # get current window display size
    try:
        rect = cv2.getWindowImageRect(WIN_NAME)
        win_w, win_h = (rect[2], rect[3]) if rect[2] > 100 and rect[3] > 100 else (raw.shape[1], raw.shape[0])
    except Exception:
        win_w, win_h = raw.shape[1], raw.shape[0]

    # scale webcam to display size, darken for contrast
    frame = cv2.resize(raw, (win_w, win_h))
    cv2.addWeighted(frame, 0.52, np.zeros_like(frame), 0.48, 0, frame)

    L    = compute_layout(win_w, win_h)
    npos = neuron_positions(L)

    if result.multi_hand_landmarks:
        draw_hand(frame, result.multi_hand_landmarks, win_w, win_h)

    newly_active   = set()
    is_drawing_now = False

    if result.multi_hand_landmarks:
        lms       = result.multi_hand_landmarks[0].landmark
        pinching  = is_pinching(lms, win_w, win_h)
        palm_open = is_open_palm(lms)

        raw_ix = lms[8].x * win_w;  raw_iy = lms[8].y * win_h
        if smooth_ix is None:
            smooth_ix, smooth_iy = raw_ix, raw_iy
        else:
            smooth_ix = SMOOTH_A * raw_ix + (1 - SMOOTH_A) * smooth_ix
            smooth_iy = SMOOTH_A * raw_iy + (1 - SMOOTH_A) * smooth_iy
        ix = int(smooth_ix);  iy = int(smooth_iy)
        gcol = (ix - L['gx']) // L['cell']
        grow = (iy - L['gy']) // L['cell']
        in_grid = 0 <= gcol < 28 and 0 <= grow < 28

        px9 = int(lms[9].x * win_w);  py9 = int(lms[9].y * win_h)
        egx = (px9 - L['gx']) // L['cell']
        egy = (py9 - L['gy']) // L['cell']
        palm_in_grid = 0 <= egx < 28 and 0 <= egy < 28

        # ── Gesture priority ────────────────────────────────────────────────
        # 1. Pinch always wins; update cooldown so erase stays suppressed ~0.3s after.
        if pinching:
            pinch_cooldown = PINCH_COOLDOWN_F
        elif pinch_cooldown > 0:
            pinch_cooldown -= 1

        # 2. Erase hold: advance only when palm clearly open AND no recent pinch.
        #    Any pinch frame resets the counter to zero (sustained-hold requirement).
        if palm_open and pinch_cooldown == 0:
            erase_hold = min(erase_hold + 1, ERASE_HOLD_F)
            if erase_hold >= ERASE_HOLD_F:
                erase_active = True
        else:
            erase_hold   = 0
            erase_active = False

        # ── Gesture actions ─────────────────────────────────────────────────
        if pinching and in_grid:
            # DRAW — always takes priority
            is_drawing_now = True
            cv2.circle(frame, (ix, iy), 18, (0, 255, 60), 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 22, (0, 85, 20),  1, cv2.LINE_AA)
            if last_gcol is not None:
                stroke_to(last_gcol, last_grow, gcol, grow, newly_active)
            else:
                paint_at(gcol, grow, newly_active)
            last_gcol, last_grow = gcol, grow

        elif erase_active and palm_in_grid:
            # ERASE — only fires after sustained hold with no pinch
            for ddy in range(-ERASE_R, ERASE_R+1):
                for ddx in range(-ERASE_R, ERASE_R+1):
                    if ddx*ddx + ddy*ddy <= ERASE_R*ERASE_R:
                        nx, ny = egx+ddx, egy+ddy
                        if 0 <= nx < 28 and 0 <= ny < 28:
                            canvas[ny, nx] = 0.0
            cv2.circle(frame, (px9, py9), 28, (0, 100, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (px9, py9), 34, (0, 50, 130),  1, cv2.LINE_AA)
            last_gcol, last_grow = None, None

        elif 0 < erase_hold < ERASE_HOLD_F and palm_in_grid:
            # CHARGING — show progress arc while building up the hold
            progress  = erase_hold / ERASE_HOLD_F
            end_angle = int(-90 + 360 * progress)
            cv2.ellipse(frame, (px9, py9), (32, 32), 0, -90, end_angle,
                        (0, 60, 160), 2, cv2.LINE_AA)
            cv2.ellipse(frame, (px9, py9), (32, 32), 0, -90, end_angle,
                        (0, 100, 255), 1, cv2.LINE_AA)
            last_gcol, last_grow = None, None

        else:
            # HOVER
            cv2.circle(frame, (ix, iy), 18, (0, 220, 220), 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 22, (0, 73, 73),   1, cv2.LINE_AA)
            last_gcol, last_grow = None, None
    else:
        smooth_ix, smooth_iy = None, None
        last_gcol, last_grow = None, None
        pinch_cooldown = 0
        erase_hold     = 0
        erase_active   = False

    for (gc, gr) in newly_active:
        if len(pulses) < MAX_PLS:
            spawn_pulse(gc, gr)
    prev_px = {(px, py) for py in range(28) for px in range(28) if canvas[py, px] > 0.1}

    # inference
    flat = preprocess_for_model(canvas)
    if flat.max() > 0.05:
        a0, a1, a2, a3 = forward(flat)
        acts       = (a0, a1, a2, a3)
        prediction = int(np.argmax(a3))
        confidence = float(a3[prediction]) * 100
        probs_str  = "  ".join(f"{i}:{a3[i]*100:.1f}%" for i in range(10))
        print(f"[pred={prediction}]  {probs_str}")
        if is_drawing_now:
            spawn_conn_particles(acts[0][INPUT_IDX])
    else:
        acts       = (np.zeros(784), np.zeros(32), np.zeros(16), np.zeros(10))
        prediction = -1
        confidence = 0.0

    # render
    draw_grid(frame, L)
    update_pulses(frame, npos, L)
    update_conn_particles(frame, npos, L)

    arr_y = L['gy'] + L['gdisp'] // 2
    cv2.arrowedLine(frame,
                    (L['gx'] + L['gdisp'] + 5, arr_y),
                    (L['nn_x1'] - 8, arr_y),
                    GRID_COL, 2, cv2.LINE_AA, tipLength=0.28)

    draw_nn(frame, acts, L, npos)
    draw_prediction_box(frame, prediction, confidence, L)

    # status banner
    if prediction >= 0 and is_drawing_now:
        banner = f"  Predicting: {prediction}   Confidence: {confidence:.1f}%  "
        bcol   = SECONDARY
    elif prediction >= 0:
        banner = f"  Last: {prediction}  ({confidence:.1f}%)  "
        bcol   = tuple(int(c * 0.65) for c in SECONDARY)
    else:
        banner = "  Pinch to draw  |  Open palm to erase  |  [C] full clear  "
        bcol   = TEXT_DIM
    bfsz = max(0.38, win_h / 1100)
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, bfsz, 2)
    bx = (win_w - tw) // 2;  by = win_h - max(th + 18, int(win_h * 0.045))
    cv2.rectangle(frame, (bx-8, by-th-8), (bx+tw+8, by+10), BG_PANEL, -1)
    cv2.rectangle(frame, (bx-8, by-th-8), (bx+tw+8, by+10), bcol, 1)
    cv2.putText(frame, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, bfsz, bcol, 2, cv2.LINE_AA)

    cv2.putText(frame, "[C] Clear   [Q/ESC] Quit",
                (win_w - max(200, int(win_w * 0.18)), 18),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.35, win_h / 2100), TEXT_DIM, 1, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if key == ord('c'):
        canvas[:] = 0
        pulses.clear()
        conn_particles.clear()
        prev_px.clear()
        acts       = (np.zeros(784), np.zeros(32), np.zeros(16), np.zeros(10))
        prediction = -1
        confidence = 0.0

cap.release()
cv2.destroyAllWindows()
