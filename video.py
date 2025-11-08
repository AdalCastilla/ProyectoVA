# demo_video.py
import cv2
import numpy as np
from preproc import (
    resize_keep_aspect, gaussian_denoise, median_denoise, bilateral_denoise,
    clahe_ycrcb, clahe_lab, gamma_correct, gray_world_white_balance, unsharp_mask
)

def apply_preproc(img, use, p):
    out = img
    if use['gauss']:    out = gaussian_denoise(out, p['g_ks'])
    if use['median']:   out = median_denoise(out, p['m_ks'])
    if use['bilateral']:out = bilateral_denoise(out, p['b_d'], p['b_sc'], p['b_ss'])
    if use['wb']:       out = gray_world_white_balance(out)
    if use['clahe_y']:  out = clahe_ycrcb(out, p['clip_y'])
    if use['clahe_l']:  out = clahe_lab(out, p['clip_l'])
    if use['gamma']:    out = gamma_correct(out, p['gamma'])
    if use['unsharp']:  out = unsharp_mask(out, p['u_ks'], p['u_amt'])
    return out

def put(frame, text, y=24, color=(0,255,0)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    use = {
        'gauss':False, 'median':False, 'bilateral':False,
        'clahe_y':False, 'clahe_l':False, 'gamma':False,
        'wb':False, 'unsharp':False
    }
    p = {
        'g_ks':3, 'm_ks':3, 'b_d':5, 'b_sc':50, 'b_ss':5,
        'clip_y':2.0, 'clip_l':2.0, 'gamma':1.0,
        'u_ks':5, 'u_amt':1.0
    }

    print("""CONTROLES:
1 Gaussian (g/G cambia kernel)
2 Median   (m/M cambia kernel)
3 Bilateral (b/B diámetro)
4 CLAHE YCrCb (c/C clip)
5 CLAHE LAB  (l/L clip)
6 Gamma   (a/A +/−)
7 White Balance
8 Unsharp (k/K kernel, u/U amount)
q salir
""")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = resize_keep_aspect(frame, 720)
        proc  = apply_preproc(frame.copy(), use, p)
        grid = np.hstack([frame, proc])

        put(grid, "Original", 24, (0,255,255))
        put(grid, "Processed", grid.shape[0]-12, (0,255,0))
        put(grid, f"ON: {[k for k,v in use.items() if v]}", 50)
        put(grid, f"g_ks={p['g_ks']} m_ks={p['m_ks']} b=({p['b_d']},{p['b_sc']},{p['b_ss']}) "
                  f"clipY={p['clip_y']} clipL={p['clip_l']} gamma={p['gamma']:.2f} "
                  f"unsharp(ks={p['u_ks']},amt={p['u_amt']})", 80)

        cv2.imshow("Preprocessing (left: original | right: processed)", grid)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break

        # toggles
        elif k == ord('1'): use['gauss'] = not use['gauss']
        elif k == ord('2'): use['median'] = not use['median']
        elif k == ord('3'): use['bilateral'] = not use['bilateral']
        elif k == ord('4'): use['clahe_y'] = not use['clahe_y']
        elif k == ord('5'): use['clahe_l'] = not use['clahe_l']
        elif k == ord('6'): use['gamma'] = not use['gamma']
        elif k == ord('7'): use['wb'] = not use['wb']
        elif k == ord('8'): use['unsharp'] = not use['unsharp']

        # params
        elif k == ord('g'): p['g_ks'] = max(1, p['g_ks'] + 2)
        elif k == ord('G'): p['g_ks'] = max(1, p['g_ks'] - 2)

        elif k == ord('m'): p['m_ks'] = max(1, p['m_ks'] + 2)
        elif k == ord('M'): p['m_ks'] = max(1, p['m_ks'] - 2)

        elif k == ord('b'): p['b_d']  = max(1, p['b_d'] + 2)
        elif k == ord('B'): p['b_d']  = max(1, p['b_d'] - 2)

        elif k == ord('c'): p['clip_y'] = float(np.clip(p['clip_y'] + 0.2, 0.5, 5.0))
        elif k == ord('C'): p['clip_y'] = float(np.clip(p['clip_y'] - 0.2, 0.5, 5.0))

        elif k == ord('l'): p['clip_l'] = float(np.clip(p['clip_l'] + 0.2, 0.5, 5.0))
        elif k == ord('L'): p['clip_l'] = float(np.clip(p['clip_l'] - 0.2, 0.5, 5.0))

        elif k == ord('a'): p['gamma'] = float(np.clip(p['gamma'] + 0.1, 0.2, 3.0))
        elif k == ord('A'): p['gamma'] = float(np.clip(p['gamma'] - 0.1, 0.2, 3.0))

        elif k == ord('k'): p['u_ks'] = max(1, p['u_ks'] + 2)
        elif k == ord('K'): p['u_ks'] = max(1, p['u_ks'] - 2)

        elif k == ord('u'): p['u_amt'] = float(np.clip(p['u_amt'] + 0.1, 0.0, 3.0))
        elif k == ord('U'): p['u_amt'] = float(np.clip(p['u_amt'] - 0.1, 0.0, 3.0))

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
