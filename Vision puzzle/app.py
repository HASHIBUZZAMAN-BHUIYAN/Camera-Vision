import cv2
import numpy as np
import mediapipe as mp
import random
import time
import os
from math import hypot

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── Palette (BGR) — matches Neural Network Visualization project ──────────────
BG_PANEL   = (22,  14,  10)   # very dark navy  RGB(10,14,22)
PRIMARY    = (240, 80,  100)  # electric indigo  RGB(100,80,240) — active/in-progress
SECONDARY  = (85,  195, 52)   # emerald green   RGB(52,195,85)  — success/correct
CURSOR_COL = (255, 55,  175)  # vivid violet    RGB(175,55,255) — active tracking point
TEXT_SOFT  = (210, 205, 220)  # near-white labels
TEXT_DIM   = (125, 115, 135)  # dim labels / hints
SLOT_EMPTY = (65,  45,  22)   # muted indigo slot border (dim PRIMARY family)

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH       = os.path.join(SCRIPT_DIR, 'captured_photo.png')

# Phase 1
RELEASE_DEBOUNCE_F = 12    # frames with 1 hand pinching before transitioning to one_corner
FREEZE_WAIT_S      = 2.0   # seconds of frozen pause before countdown starts

# Phase 2
PICK_HOLD_F        = 4     # sustained-pinch frames before picking up a piece
RELEASE_HOLD_F     = 6     # non-pinch frames before dropping a piece
SMOOTH_A           = 0.35  # EMA alpha for pinch cursor smoothing

# ── Mutable app state ─────────────────────────────────────────────────────────
app_state = 'capture'   # 'capture' | 'puzzle' | 'solved'

# Phase 1 — capture state machine
cap_state = {
    'mode':          'idle',  # 'idle' | 'one_corner' | 'live_box' | 'frozen'
    'corner_a':      None,    # (x, y) pinch midpoint of hand A
    'corner_b':      None,    # (x, y) pinch midpoint of hand B
    'last_live_box': None,    # (x0, y0, x1, y1) — last valid two-hand rectangle
    'release_frames':0,       # counts up while only 1 hand pinching in live_box
    'freeze_t':      None,    # time.time() when freeze started
}

# Phase 2
pieces_img    = []    # list[9] of BGR ndarrays
slot_to_piece = {}    # slot_idx → piece_idx | None
piece_to_slot = {}    # piece_idx → slot_idx | None
piece_placed  = []    # [bool] * 9

drag = {
    'active':         False,
    'piece_idx':      None,
    'origin_slot':    None,
    'pos':            (0, 0),
    'pick_frames':    0,
    'release_frames': 0,
}

solved_t  = None
smooth_mx = None
smooth_my = None

# ── Shared helpers ────────────────────────────────────────────────────────────
def is_pinching(lms, w, h):
    tx, ty = lms[4].x * w, lms[4].y * h
    ix, iy = lms[8].x * w, lms[8].y * h
    return hypot(tx - ix, ty - iy) < w * 0.035

def raw_pinch_mid(lms, w, h):
    return (
        int((lms[4].x + lms[8].x) / 2 * w),
        int((lms[4].y + lms[8].y) / 2 * h),
    )

def draw_hand(frame, lms_list, w, h):
    skel_dim = tuple(c // 4 for c in SECONDARY)   # dim emerald for joint fill
    skel_hi  = tuple(min(255, int(c * 1.25)) for c in SECONDARY)  # bright emerald highlight
    for hlms in lms_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hlms.landmark]
        for a_i, b_i in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, pts[a_i], pts[b_i], SECONDARY, 3, cv2.LINE_AA)
            cv2.line(frame, pts[a_i], pts[b_i], skel_hi,   1, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(frame, (px, py), 6, skel_dim,  -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, SECONDARY, -1, cv2.LINE_AA)

def draw_status(frame, text, color, win_w, win_h):
    # Scale font so text always fits horizontally with padding
    pad  = max(6, int(win_h * 0.012))
    fsz  = max(0.30, win_h / 1100)
    max_tw = win_w - 4 * pad
    while fsz > 0.28:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fsz, 1)
        if tw <= max_tw:
            break
        fsz -= 0.02
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fsz, 1)
    bar_h = th + 2 * pad
    bx    = (win_w - tw) // 2
    by    = win_h - pad - th          # baseline of text
    # Keep entirely within frame
    by    = min(by, win_h - th - 2)
    panel_y0 = max(0, by - th - pad)
    panel_y1 = min(win_h - 1, by + pad)
    ov = frame.copy()
    cv2.rectangle(ov, (0, panel_y0), (win_w, panel_y1), BG_PANEL, -1)
    cv2.addWeighted(frame, 0.28, ov, 0.72, 0, frame)
    cv2.rectangle(frame, (0, panel_y0), (win_w, panel_y1), tuple(c//3 for c in color), 1)
    cv2.putText(frame, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, fsz, color, 1, cv2.LINE_AA)

def glow_rect(frame, x0, y0, x1, y1, color, thick=2):
    dim = tuple(c // 6 for c in color)
    cv2.rectangle(frame, (x0-3, y0-3), (x1+3, y1+3), dim, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)

def blit_safe(frame, img, dx, dy):
    fh, fw = frame.shape[:2]
    ih, iw = img.shape[:2]
    x0 = max(0, dx);  y0 = max(0, dy)
    x1 = min(fw, dx + iw);  y1 = min(fh, dy + ih)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = img[y0-dy:y0-dy+(y1-y0), x0-dx:x0-dx+(x1-x0)]

# ── Phase 1 helpers ───────────────────────────────────────────────────────────
def get_pinching_hands(all_hands, win_w, win_h):
    """Return list of (mx, my) pinch midpoints for each currently-pinching hand."""
    pts = []
    if not all_hands:
        return pts
    for hlms in all_hands:
        lms = hlms.landmark
        if is_pinching(lms, win_w, win_h):
            pts.append(raw_pinch_mid(lms, win_w, win_h))
    return pts

def box_from_corners(pt_a, pt_b):
    """Normalise two corner points to (x0, y0, x1, y1) with x0≤x1, y0≤y1."""
    return (min(pt_a[0], pt_b[0]), min(pt_a[1], pt_b[1]),
            max(pt_a[0], pt_b[0]), max(pt_a[1], pt_b[1]))

# ── Phase 1 — capture ─────────────────────────────────────────────────────────
def do_capture(raw_flip, bx0, by0, bx1, by1, win_w, win_h):
    global app_state, pieces_img, slot_to_piece, piece_to_slot, piece_placed, solved_t

    rh, rw = raw_flip.shape[:2]
    sx = rw / win_w;  sy = rh / win_h
    rx0 = max(0, int(bx0 * sx));  ry0 = max(0, int(by0 * sy))
    rx1 = min(rw, int(bx1 * sx)); ry1 = min(rh, int(by1 * sy))

    if rx1 - rx0 < 60 or ry1 - ry0 < 60:
        return

    region = raw_flip[ry0:ry1, rx0:rx1].copy()
    h_r, w_r = region.shape[:2]
    cv2.imwrite(CAPTURE_PATH, region)

    # Slice 3×3 (handles any aspect ratio)
    imgs = []
    for r in range(3):
        for c in range(3):
            ys = int(h_r * r / 3);  ye = int(h_r * (r+1) / 3) if r < 2 else h_r
            xs = int(w_r * c / 3);  xe = int(w_r * (c+1) / 3) if c < 2 else w_r
            imgs.append(region[ys:ye, xs:xe].copy())
    pieces_img = imgs

    order = list(range(9))
    random.shuffle(order)
    slot_to_piece = {i: order[i] for i in range(9)}
    piece_to_slot = {order[i]: i for i in range(9)}
    piece_placed  = [False] * 9
    drag.update({'active': False, 'piece_idx': None, 'origin_slot': None,
                 'pos': (0, 0), 'pick_frames': 0, 'release_frames': 0})
    solved_t  = None
    app_state = 'puzzle'

def handle_capture_phase(frame, all_hands, win_w, win_h, raw_flip):
    global cap_state

    pinch_pts = get_pinching_hands(all_hands, win_w, win_h)
    n_pinch   = len(pinch_pts)
    mode      = cap_state['mode']

    # ── State transitions ─────────────────────────────────────────────────────
    if mode == 'idle':
        if n_pinch == 1:
            cap_state['mode']     = 'one_corner'
            cap_state['corner_a'] = pinch_pts[0]
        elif n_pinch == 2:
            cap_state['mode']          = 'live_box'
            cap_state['corner_a']      = pinch_pts[0]
            cap_state['corner_b']      = pinch_pts[1]
            cap_state['last_live_box'] = box_from_corners(pinch_pts[0], pinch_pts[1])
            cap_state['release_frames'] = 0

    elif mode == 'one_corner':
        if n_pinch == 0:
            cap_state['mode']     = 'idle'
            cap_state['corner_a'] = None
        elif n_pinch == 1:
            cap_state['corner_a'] = pinch_pts[0]
        else:  # 2
            cap_state['mode']          = 'live_box'
            cap_state['corner_a']      = pinch_pts[0]
            cap_state['corner_b']      = pinch_pts[1]
            cap_state['last_live_box'] = box_from_corners(pinch_pts[0], pinch_pts[1])
            cap_state['release_frames'] = 0

    elif mode == 'live_box':
        if n_pinch == 2:
            cap_state['corner_a']       = pinch_pts[0]
            cap_state['corner_b']       = pinch_pts[1]
            cap_state['last_live_box']  = box_from_corners(pinch_pts[0], pinch_pts[1])
            cap_state['release_frames'] = 0
        elif n_pinch == 0:
            # Both hands released — freeze immediately
            cap_state['mode']           = 'frozen'
            cap_state['freeze_t']       = time.time()
            cap_state['release_frames'] = 0
        else:  # n_pinch == 1 — one hand released, debounce
            cap_state['release_frames'] += 1
            if cap_state['release_frames'] >= RELEASE_DEBOUNCE_F:
                # Remaining hand stayed pinching too long — treat as one_corner
                cap_state['mode']           = 'one_corner'
                cap_state['corner_a']       = pinch_pts[0]
                cap_state['corner_b']       = None
                cap_state['release_frames'] = 0

    elif mode == 'frozen':
        if n_pinch > 0:
            # Any pinch → cancel freeze, return to live adjustment
            cap_state['freeze_t']       = None
            cap_state['release_frames'] = 0
            if n_pinch == 2:
                cap_state['mode']          = 'live_box'
                cap_state['corner_a']      = pinch_pts[0]
                cap_state['corner_b']      = pinch_pts[1]
                cap_state['last_live_box'] = box_from_corners(pinch_pts[0], pinch_pts[1])
            else:
                cap_state['mode']     = 'one_corner'
                cap_state['corner_a'] = pinch_pts[0]
                cap_state['corner_b'] = None
        else:
            # Count time; trigger capture after wait + 2 countdown digits
            elapsed = time.time() - cap_state['freeze_t']
            if elapsed >= FREEZE_WAIT_S + 2.0:
                x0, y0, x1, y1 = cap_state['last_live_box']
                do_capture(raw_flip, x0, y0, x1, y1, win_w, win_h)
                cap_state.update({'mode': 'idle', 'corner_a': None, 'corner_b': None,
                                  'last_live_box': None, 'freeze_t': None,
                                  'release_frames': 0})
                return

    # ── Draw ─────────────────────────────────────────────────────────────────
    mode = cap_state['mode']  # re-read after transitions

    if mode == 'idle':
        draw_status(frame,
                    "Pinch with both hands — one corner each — to frame the shot",
                    TEXT_DIM, win_w, win_h)

    elif mode == 'one_corner':
        mx, my = cap_state['corner_a']
        cv2.circle(frame, (mx, my), 10, CURSOR_COL, -1, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), 16, tuple(c//3 for c in CURSOR_COL), 1, cv2.LINE_AA)
        draw_status(frame, "Corner set — now pinch with your other hand to stretch the frame",
                    PRIMARY, win_w, win_h)

    elif mode == 'live_box':
        box = cap_state['last_live_box']
        if box:
            x0, y0, x1, y1 = box
            glow_rect(frame, x0, y0, x1, y1, PRIMARY, 3)
            # Corner dots
            for pt in (cap_state['corner_a'], cap_state['corner_b']):
                if pt:
                    cv2.circle(frame, pt, 9, CURSOR_COL, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 14, tuple(c//4 for c in CURSOR_COL), 1, cv2.LINE_AA)
            # Inline hint in box centre
            cx_ = (x0 + x1) // 2;  cy_ = (y0 + y1) // 2
            hint = "Release both hands to freeze"
            hfsz = max(0.30, min(x1-x0, y1-y0) / 420)
            (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hfsz, 1)
            cv2.putText(frame, hint, (cx_ - tw//2, cy_ + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, hfsz,
                        tuple(c//2 for c in TEXT_SOFT), 1, cv2.LINE_AA)
        draw_status(frame, "Frame box live — adjust then release both hands to freeze",
                    TEXT_SOFT, win_w, win_h)

    elif mode == 'frozen':
        x0, y0, x1, y1 = cap_state['last_live_box']
        cx_ = (x0 + x1) // 2;  cy_ = (y0 + y1) // 2
        elapsed = time.time() - cap_state['freeze_t']

        if elapsed < FREEZE_WAIT_S:
            # Waiting — PRIMARY pulsing border + progress bar
            pulse = 0.65 + 0.35 * np.sin(elapsed * 7)
            col   = tuple(int(c * pulse) for c in PRIMARY)
            glow_rect(frame, x0, y0, x1, y1, col, 3)
            bar_w  = x1 - x0
            fill   = elapsed / FREEZE_WAIT_S
            bar_y0 = y1 + 8
            bar_y1 = y1 + max(6, int(win_h * 0.008))
            if bar_y1 < win_h:
                cv2.rectangle(frame, (x0, bar_y0), (x0+bar_w, bar_y1),
                              tuple(c//4 for c in PRIMARY), -1)
                cv2.rectangle(frame, (x0, bar_y0), (x0+int(bar_w*fill), bar_y1),
                              PRIMARY, -1)
            draw_status(frame, "Frozen — preparing to capture...", PRIMARY, win_w, win_h)

        else:
            # Countdown 2 → 1 — PRIMARY box, SECONDARY digit
            cd_elapsed = elapsed - FREEZE_WAIT_S
            digit = max(1, 2 - int(cd_elapsed))
            glow_rect(frame, x0, y0, x1, y1, PRIMARY, 3)
            box_min = min(x1-x0, y1-y0)
            fsz  = max(1.0, box_min / 120)
            thk  = max(2, int(box_min / 70))
            d    = str(digit)
            (tw, th), _ = cv2.getTextSize(d, cv2.FONT_HERSHEY_SIMPLEX, fsz, thk)
            cv2.putText(frame, d, (cx_-tw//2+2, cy_+th//2+2),
                        cv2.FONT_HERSHEY_SIMPLEX, fsz,
                        tuple(c//5 for c in SECONDARY), thk, cv2.LINE_AA)
            cv2.putText(frame, d, (cx_-tw//2, cy_+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fsz, SECONDARY, thk, cv2.LINE_AA)
            draw_status(frame, "Capturing in...", PRIMARY, win_w, win_h)

# ── Phase 2 ───────────────────────────────────────────────────────────────────
def compute_grid(win_w, win_h):
    grid_size  = min(win_w, win_h) * 0.62
    gap        = max(4, int(grid_size * 0.016))
    piece_size = int((grid_size - 4 * gap) / 3)
    total      = 3 * piece_size + 4 * gap
    ox         = (win_w - total) // 2
    oy         = (win_h - total) // 2
    rects, centers = [], []
    for row in range(3):
        for col in range(3):
            x = ox + gap + col * (piece_size + gap)
            y = oy + gap + row * (piece_size + gap)
            rects.append((x, y, piece_size, piece_size))
            centers.append((x + piece_size // 2, y + piece_size // 2))
    return {'ps': piece_size, 'rects': rects, 'centers': centers}

def drop_piece(G):
    global drag, slot_to_piece, piece_to_slot, piece_placed, app_state, solved_t

    pidx   = drag['piece_idx']
    pos    = drag['pos']
    origin = drag['origin_slot']
    ps     = G['ps']

    best_s, best_d = None, float('inf')
    for s in range(9):
        cx_, cy_ = G['centers'][s]
        d = hypot(pos[0]-cx_, pos[1]-cy_)
        if d < best_d:
            best_d, best_s = d, s

    if best_d > ps * 0.8:
        target = origin
    else:
        occupant = slot_to_piece.get(best_s)
        if occupant is not None and piece_placed[occupant]:
            target = origin
        elif occupant is not None:
            slot_to_piece[origin] = occupant
            piece_to_slot[occupant] = origin
            target = best_s
        else:
            target = best_s

    slot_to_piece[target] = pidx
    piece_to_slot[pidx]   = target
    if target == pidx:
        piece_placed[pidx] = True

    drag.update({'active': False, 'piece_idx': None, 'origin_slot': None,
                 'pos': (0, 0), 'pick_frames': 0, 'release_frames': 0})

    if all(piece_placed):
        app_state = 'solved'
        solved_t  = time.time()

def handle_puzzle_phase(frame, lms, spmx, spmy, win_w, win_h):
    global drag, app_state, solved_t

    if not pieces_img:
        return

    G  = compute_grid(win_w, win_h)
    ps = G['ps']

    pinching = lms is not None and is_pinching(lms, win_w, win_h)
    pmx, pmy = (spmx, spmy) if lms is not None else (0, 0)

    # ── Gesture state machine ─────────────────────────────────────────────────
    if not drag['active']:
        if pinching:
            drag['pick_frames'] += 1
            if drag['pick_frames'] >= PICK_HOLD_F:
                for s in range(9):
                    pidx = slot_to_piece.get(s)
                    if pidx is None or piece_placed[pidx]:
                        continue
                    x, y, w_, h_ = G['rects'][s]
                    if x <= pmx <= x+w_ and y <= pmy <= y+h_:
                        drag.update({'active': True, 'piece_idx': pidx,
                                     'origin_slot': s, 'pos': (pmx, pmy),
                                     'pick_frames': 0, 'release_frames': 0})
                        slot_to_piece[s]    = None
                        piece_to_slot[pidx] = None
                        break
        else:
            drag['pick_frames'] = max(0, drag['pick_frames'] - 1)
    else:
        if lms is not None:
            drag['pos'] = (pmx, pmy)
        if not pinching:
            drag['release_frames'] += 1
            if drag['release_frames'] >= RELEASE_HOLD_F:
                drop_piece(G)
        else:
            drag['release_frames'] = 0

    # Nearest slot for drop-target highlight
    nearest = None
    if drag['active']:
        bd = float('inf')
        for s in range(9):
            cx_, cy_ = G['centers'][s]
            d = hypot(drag['pos'][0]-cx_, drag['pos'][1]-cy_)
            if d < bd:
                bd, nearest = d, s
        if bd > ps * 0.8:
            nearest = None

    # ── Draw slots and resting pieces ─────────────────────────────────────────
    for s in range(9):
        x, y, w_, h_ = G['rects'][s]
        pidx = slot_to_piece.get(s)

        if pidx is not None and piece_placed[pidx]:
            pimg = cv2.resize(pieces_img[pidx], (w_, h_))
            ov   = pimg.copy()
            cv2.rectangle(ov, (0,0), (w_-1,h_-1), SECONDARY, -1)
            cv2.addWeighted(pimg, 0.93, ov, 0.07, 0, pimg)
            blit_safe(frame, pimg, x, y)
            cv2.rectangle(frame, (x-1, y-1), (x+w_+1, y+h_+1),
                          tuple(c//4 for c in SECONDARY), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w_, y+h_), SECONDARY, 2, cv2.LINE_AA)

        elif pidx is not None:
            pimg = cv2.resize(pieces_img[pidx], (w_, h_))
            blit_safe(frame, pimg, x, y)
            col  = PRIMARY if s == nearest else SLOT_EMPTY
            cv2.rectangle(frame, (x-1, y-1), (x+w_+1, y+h_+1),
                          tuple(c//4 for c in col), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w_, y+h_), col, 2, cv2.LINE_AA)

        else:
            ov = frame.copy()
            cv2.rectangle(ov, (x, y), (x+w_, y+h_), BG_PANEL, -1)
            cv2.addWeighted(frame, 0.55, ov, 0.45, 0, frame)
            col  = PRIMARY if s == nearest else SLOT_EMPTY
            thk  = 3 if s == nearest else 1
            cv2.rectangle(frame, (x, y), (x+w_, y+h_), col, thk, cv2.LINE_AA)

    # ── Dragged piece on top ──────────────────────────────────────────────────
    if drag['active']:
        pidx   = drag['piece_idx']
        dsz    = int(ps * 1.08)
        pimg   = cv2.resize(pieces_img[pidx], (dsz, dsz))
        dx     = max(0, min(win_w - dsz, drag['pos'][0] - dsz // 2))
        dy     = max(0, min(win_h - dsz, drag['pos'][1] - dsz // 2))
        blit_safe(frame, pimg, dx, dy)
        ex = min(win_w-1, dx+dsz);  ey = min(win_h-1, dy+dsz)
        cv2.rectangle(frame, (dx-1, dy-1), (ex+1, ey+1),
                      tuple(c//4 for c in PRIMARY), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (dx, dy), (ex, ey), PRIMARY, 3, cv2.LINE_AA)

    elif lms is not None:
        cv2.circle(frame, (pmx, pmy), 11, tuple(c//4 for c in CURSOR_COL), -1, cv2.LINE_AA)
        cv2.circle(frame, (pmx, pmy),  7, CURSOR_COL, -1, cv2.LINE_AA)
        cv2.circle(frame, (pmx, pmy),  3, (230, 230, 240), -1, cv2.LINE_AA)

    # ── Progress & hints ──────────────────────────────────────────────────────
    n_placed = sum(piece_placed)
    pfsz = max(0.45, win_h / 1400)
    cv2.putText(frame, f"{n_placed} / 9 pieces placed correctly",
                (max(10, int(win_w*0.015)), max(28, int(win_h*0.05))),
                cv2.FONT_HERSHEY_SIMPLEX, pfsz, TEXT_SOFT, 1, cv2.LINE_AA)

    hint = "[R] Restart   [Q] Quit"
    hfsz = max(0.32, win_h / 1900)
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hfsz, 1)
    cv2.putText(frame, hint,
                (win_w - hw - max(8, int(win_w*0.01)), max(22, int(win_h*0.04))),
                cv2.FONT_HERSHEY_SIMPLEX, hfsz, TEXT_DIM, 1, cv2.LINE_AA)

    if drag['active']:
        draw_status(frame, "Release pinch to drop", PRIMARY, win_w, win_h)
    elif lms is None:
        draw_status(frame, "Show your hand — pinch and hold a piece to move it", TEXT_DIM, win_w, win_h)
    else:
        draw_status(frame, "Pinch + hold over a piece to pick it up", TEXT_SOFT, win_w, win_h)

    if all(piece_placed) and app_state == 'puzzle':
        app_state = 'solved'
        solved_t  = time.time()

def draw_solved_overlay(frame, win_w, win_h):
    if solved_t is None:
        return
    elapsed = time.time() - solved_t
    alpha   = min(0.70, elapsed * 0.6)
    pulse   = 0.82 + 0.18 * np.sin(elapsed * 3.5)

    dark = np.zeros_like(frame)
    dark[:] = BG_PANEL
    cv2.addWeighted(frame, 1.0 - alpha, dark, alpha, 0, frame)

    text  = "SOLVED!"
    fsz   = max(1.8, win_h / 290)
    thick = max(3, int(win_h / 185))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fsz, thick)
    tx = (win_w - tw) // 2
    ty = int(win_h * 0.46) + th // 2

    col = tuple(min(255, int(c * pulse)) for c in SECONDARY)
    cv2.putText(frame, text, (tx+3, ty+3), cv2.FONT_HERSHEY_SIMPLEX, fsz,
                tuple(c//5 for c in SECONDARY), thick, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fsz,
                col, thick, cv2.LINE_AA)

    sub  = "[R] Play again   [Q] Quit"
    sfsz = max(0.42, win_h / 1450)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, sfsz, 1)
    cv2.putText(frame, sub, ((win_w-sw)//2, ty + int(win_h*0.11)),
                cv2.FONT_HERSHEY_SIMPLEX, sfsz, TEXT_DIM, 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────
WIN_NAME = "Vision Puzzle"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

print("Vision Puzzle — ready.")
print("  Phase 1 : Pinch both hands to define a frame box, release both to freeze,")
print("            then wait for the countdown to capture.")
print("  Phase 2 : Pinch + hold over a piece to drag it into position.")
print("  [R] restart capture   [Q / ESC] quit")

while True:
    ret, raw = cap.read()
    if not ret:
        break
    raw_flip = cv2.flip(raw, 1)

    rgb    = cv2.cvtColor(raw_flip, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    try:
        rect  = cv2.getWindowImageRect(WIN_NAME)
        win_w = rect[2] if rect[2] > 100 else raw.shape[1]
        win_h = rect[3] if rect[3] > 100 else raw.shape[0]
    except Exception:
        win_w, win_h = raw.shape[1], raw.shape[0]

    # Darkened, blue-tinted webcam background
    frame = cv2.resize(raw_flip, (win_w, win_h))
    tint  = np.empty_like(frame);  tint[:] = BG_PANEL
    cv2.addWeighted(frame, 0.50, tint, 0.50, 0, frame)

    all_hands = result.multi_hand_landmarks   # list of 0–2 hand landmarks
    lms       = all_hands[0].landmark if all_hands else None  # first hand (puzzle phase)

    # EMA-smoothed pinch midpoint for the puzzle phase (single hand)
    if lms is not None:
        rx, ry = raw_pinch_mid(lms, win_w, win_h)
        if smooth_mx is None:
            smooth_mx = float(rx);  smooth_my = float(ry)
        else:
            smooth_mx = SMOOTH_A * rx + (1 - SMOOTH_A) * smooth_mx
            smooth_my = SMOOTH_A * ry + (1 - SMOOTH_A) * smooth_my
        spmx, spmy = int(smooth_mx), int(smooth_my)
    else:
        smooth_mx = smooth_my = None
        spmx, spmy = 0, 0

    if all_hands:
        draw_hand(frame, all_hands, win_w, win_h)

    if app_state == 'capture':
        handle_capture_phase(frame, all_hands, win_w, win_h, raw_flip)
    elif app_state in ('puzzle', 'solved'):
        handle_puzzle_phase(frame, lms, spmx, spmy, win_w, win_h)
        if app_state == 'solved':
            draw_solved_overlay(frame, win_w, win_h)

    cv2.imshow(WIN_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), 27):
        break
    if key == ord('r'):
        app_state = 'capture'
        cap_state.update({'mode': 'idle', 'corner_a': None, 'corner_b': None,
                          'last_live_box': None, 'release_frames': 0, 'freeze_t': None})
        solved_t        = None
        smooth_mx = smooth_my = None
        drag.update({'active': False, 'piece_idx': None, 'origin_slot': None,
                     'pos': (0, 0), 'pick_frames': 0, 'release_frames': 0})

cap.release()
cv2.destroyAllWindows()
