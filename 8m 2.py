#!/usr/bin/env python3
"""
clip8m.py — 初期位置から 8m 先の ARマーカーまで飛び、クリップで真上に合わせて正確に着陸する (同一地点での複数回計測・認識強化版)
=====================================================================
hula_jp.py の HulaController を使う。同じフォルダに置くこと。
"""
import argparse
import math

from hula_jp import HulaController, FakeApi, DroneError, S, world_to_body, body_to_world, install_stop_handlers

LEGS_CM = (250, 250, 300)  # 250cm刻みで8m（APIの300cm上限対策）
FWD_SPEED_CM_S = 100
FIX_TOL_CM = 25
SEARCH_FB_CM = 60
SEARCH_FB_MAX_CM = 180
SEARCH_SPEED_CM_S = 60
COARSE_ALT_CM = 100
COARSE_TIMEOUT_S = 15
FINE_TIMEOUT_S = 15
ATTEMPT_TIME_S = 120
LAND_RESERVE_S = 18

# ★ 追加: 同一地点でのマーカー認識リトライ設定
RECOG_RETRIES = 5          # 1回の位置でマーカー認識を何回試すか
RECOG_INTERVAL_S = 0.2     # リトライ間のウェイト（秒）


def build_parser():
    p = argparse.ArgumentParser(description="8m 先の ARマーカーまで飛んで正確に着陸する")
    p.add_argument("--qr", type=int, default=1, help="着陸するマーカーの番号（既定 1）")
    p.add_argument("--legs", type=int, nargs="+", default=list(LEGS_CM), help="前進の区切り cm")
    p.add_argument("--dry", action="store_true", help="機体に接続せず、偽の機体で練習する")
    p.add_argument("--error", type=float, default=0.0, help="--dry のとき、マーカーを前後に何 cm ずらすか")
    p.add_argument("--seed", type=int, default=None, help="--dry の乱数シード")
    p.add_argument("--yaw0", type=float, default=-35.0, help="--dry の yaw")
    p.add_argument("--abs-frame", action="store_true", help="--dry の基準")
    return p


def connect_drone():
    try:
        import pyhula
    except ImportError:
        raise DroneError("pyhula が見つかりません。")
    api = pyhula.UserApi()
    if not api.connect():
        raise DroneError("機体に接続できませんでした。")
    return api


def patch_controller_api(c):
    """実機APIの差異や戻り値のゆれ（辞書か数値か）を完全に吸収するパッチ"""
    is_fake = isinstance(c.api, FakeApi)

    # 1. 移動コマンドの第2引数（LEDパラメータ誤認）対策
    orig_api_call = c._api_call
    move_cmds = {"single_fly_forward", "single_fly_back", "single_fly_left", "single_fly_right", "single_fly_up", "single_fly_down"}

    def safe_api_call(func_name, *args):
        if not is_fake and func_name in move_cmds and len(args) >= 2 and isinstance(args[1], (int, float)):
            try:
                return orig_api_call(func_name, args[0])
            except DroneError:
                return orig_api_call(func_name, *args)
        return orig_api_call(func_name, *args)

    c._api_call = safe_api_call

    # 2. QR認識のロバスト化（複数の公式API名と戻り値形式に完全対応）
    def patched_see_marker(qr):
        r = None
        # (A) single_fly_Optical_flow_recognition
        if hasattr(c.api, "single_fly_Optical_flow_recognition"):
            try:
                res = c._api_call("single_fly_Optical_flow_recognition", qr, S.QR_SIZE_CM)
                if isinstance(res, dict) and (res.get("result") or res.get("result") == 1):
                    r = res
            except DroneError:
                pass

        # (B) single_fly_recognition_Qrcode(mode=0, qr_id)
        if r is None and hasattr(c.api, "single_fly_recognition_Qrcode"):
            try:
                res2 = c._api_call("single_fly_recognition_Qrcode", 0, qr)
                if isinstance(res2, dict) and (res2.get("result") or res2.get("result") == 1):
                    r = res2
                elif res2 == 1 or res2 is True:
                    r = {"result": True, "x": 0.0, "y": 0.0}
            except DroneError:
                pass

        if not isinstance(r, dict) or not (r.get("result") or r.get("result") == 1):
            return None

        off = float(r.get("x", 0)) * S.MARKER_SIGN, float(r.get("y", 0)) * S.MARKER_SIGN
        try:
            x, y, _ = c.xyz()
            wx, wy = body_to_world(off[0], off[1], c.heading())
            c.marker_xy[qr] = (x + wx, y + wy)
        except Exception:
            pass
        return off

    c.see_marker = patched_see_marker

    # ★ 追加: 同じ場所で複数回（RECOG_RETRIES回）連続でマーカー認識を試みるラッパー
    def robust_see_marker(qr):
        for attempt_idx in range(RECOG_RETRIES):
            res = c.see_marker(qr)
            if res is not None:
                if attempt_idx > 0:
                    print(f"    (※ 認識リトライ {attempt_idx + 1}回目でマーカーを検出成功)")
                return res
            if attempt_idx < RECOG_RETRIES - 1:
                c.sleep(RECOG_INTERVAL_S)
        return None

    c.robust_see_marker = robust_see_marker

    # 3. 位置合わせAPIのフォールバック
    def patched_call_align_api(qr):
        t0 = c.now()
        ok = False
        if hasattr(c.api, "single_fly_Optical_flow_alignment"):
            try:
                res = c._api_call("single_fly_Optical_flow_alignment", qr, S.QR_SIZE_CM, 0)
                ok = res.get("result", False) if isinstance(res, dict) else (res == 1 or res is True)
            except DroneError:
                ok = False
        if not ok and hasattr(c.api, "single_fly_Qrcode_align"):
            try:
                res = c._api_call("single_fly_Qrcode_align", 0, qr)
                ok = res.get("result", False) if isinstance(res, dict) else (res == 1 or res is True)
            except DroneError:
                ok = False
        c.cost_s["align_api"] = c.now() - t0
        c.measured.add("align_api")
        return ok

    c._call_align_api = patched_call_align_api


def split_safe_legs(legs, max_step=250):
    out = []
    for leg in legs:
        rem = float(leg)
        while rem >= S.MIN_MANUAL_MOVE_CM:
            step = min(rem, float(max_step))
            if 0 < rem - step < S.MIN_MANUAL_MOVE_CM:
                step = rem - S.MIN_MANUAL_MOVE_CM
            out.append(int(round(step)))
            rem -= step
    return out


def show(c, label):
    x, y, z = c.xyz()
    print("%s  座標 (%.0f, %+.0f)  高さ %.0fcm  向き %+.0f°  電池 %d%%"
          % (label, x, y, z, c.heading(), c.battery()))


def forward_only(c, d, speed=S.FINE_SPEED_CM_S):
    if abs(d) < S.MIN_MANUAL_MOVE_CM:
        return
    c._axis("single_fly_forward", "single_fly_back", d, speed)


def advance(c, legs):
    safe_legs = split_safe_legs(legs, max_step=250)
    x0, y0, _ = c.raw_xyz()

    for i, leg in enumerate(safe_legs):
        c.fix_heading()
        h = c.heading()
        xa, ya, _ = c.raw_xyz()
        print("前に %dcm 進みます（%d/%d）" % (leg, i + 1, len(safe_legs)))
        c._api_call("single_fly_forward", leg, FWD_SPEED_CM_S)
        xb, yb, _ = c.raw_xyz()
        moved = math.hypot(xb - xa, yb - ya)

        if moved < 0.6 * leg:
            rest = leg - moved
            print("  ⚠ %.0fcm しか進んでいません。残り %.0fcm を再送します" % (moved, rest))
            c.sleep(0.5)
            forward_only(c, rest, FWD_SPEED_CM_S)
            xb, yb, _ = c.raw_xyz()

        total_from_start = math.hypot(xb - x0, yb - y0)
        if not c.calibrated and total_from_start >= S.CALIB_FWD_CM:
            c.calibrate_from_forward(x0, y0, xb, yb, h)
        show(c, "  ")

    target = float(sum(legs))
    x, y, _ = c.xyz()
    fwd, left = world_to_body(target - x, 0.0 - y, c.heading())
    if abs(fwd) >= FIX_TOL_CM:
        print("  前後に %+.0fcm ずれているので直します" % fwd)
        forward_only(c, fwd, SEARCH_SPEED_CM_S)
        show(c, "  ")

    x, y, _ = c.xyz()
    _, left = world_to_body(target - x, 0.0 - y, c.heading())
    if c.calibrated and abs(left) >= 45:
        print("  左右に %+.0fcm 大きく流れているため中央へ戻します" % (-left))
        c._axis("single_fly_left", "single_fly_right", left, SEARCH_SPEED_CM_S)
        show(c, "  ")


def find_marker_fb(c, qr, deadline):
    """マーカー探索：必ずホバリング状態で停止してから、複数回計測を行いながらカメラを覗く"""
    c.fix_heading()
    c._go_alt(min(S.SEARCH_ALT_CM, c.max_alt()))
    c.sleep(0.8)  # 停止後のカメラ安定待ち
    
    if c.robust_see_marker(qr) is not None:
        print("    マーカー発見（高度を上げた時点で発見）")
        return True

    x0, y0, _ = c.xyz()
    if c.calibrated and abs(y0) >= 40:
        _, left = world_to_body(0.0, -y0, c.heading())
        if abs(left) >= S.MIN_MANUAL_MOVE_CM:
            c._axis("single_fly_left", "single_fly_right", left, SEARCH_SPEED_CM_S)

    steps = list(range(SEARCH_FB_CM, SEARCH_FB_MAX_CM + 1, SEARCH_FB_CM))
    offsets = [0] + steps + [-k for k in steps]
    for d in offsets:
        if deadline - c.now() < LAND_RESERVE_S + 8:
            return False
        gx = min(max(x0 + d, 0.0), S.FIELD_X_MAX_CM - S.GEOFENCE_MARGIN_CM)
        x, _, _ = c.xyz()
        fwd, _ = world_to_body(gx - x, 0.0, c.heading())
        if abs(fwd) >= S.MIN_MANUAL_MOVE_CM:
            forward_only(c, fwd, SEARCH_SPEED_CM_S)
        
        c.sleep(0.5)  # 移動停止後のブレ収束待ち
        if c.robust_see_marker(qr) is not None:
            print("    マーカー発見（前後 %+dcm）" % d)
            return True

    high_alt = min(190, c.max_alt())
    if high_alt > S.SEARCH_ALT_CM and deadline - c.now() >= LAND_RESERVE_S + 10:
        print("    高度を %dcm に上げて再探索します" % high_alt)
        c._go_alt(high_alt)
        c.sleep(1.0)
        if c.robust_see_marker(qr) is not None:
            print("    高高度 (%dcm) でマーカー発見" % high_alt)
            return True
    return False


def clip_and_land(c, qr, tx, ty, deadline):
    if not getattr(c, "_patched_by_clip8m", False):
        patch_controller_api(c)
        c._patched_by_clip8m = True

    attempt = 0
    while True:
        attempt += 1
        left_s = deadline - c.now()
        if left_s < LAND_RESERVE_S + 5:
            print("  時間切れのため、その場に着陸します")
            break

        c.sleep(0.5)
        # ★ ここでも robust_see_marker を使用して複数回チェック
        if c.robust_see_marker(qr) is None:
            print("  [クリップ %d回目] マーカー未検出（複数回確認） → 探索開始" % attempt)
            if not find_marker_fb(c, qr, deadline):
                print("    見つからないためその場に着陸します")
                break

        left_s = deadline - c.now()
        print("  [クリップ %d回目] 粗く合わせる（高さ %dcm）" % (attempt, COARSE_ALT_CM))
        coarse = c.clip(qr, COARSE_ALT_CM, min(COARSE_TIMEOUT_S, left_s - LAND_RESERVE_S), use_align=False)
        
        if not coarse:
            bx, by = c.marker_xy.get(qr, (tx, ty))
            x, y, _ = c.xyz()
            fwd, left = world_to_body(bx - x, by - y, c.heading())
            print("    粗寄せ失敗 → (%+.0f, %+.0f)cm 戻って再試行" % (fwd, left))
            c._move_manual(fwd, left, speed=SEARCH_SPEED_CM_S)
            continue

        left_s = deadline - c.now()
        if left_s < LAND_RESERVE_S + 5:
            c.land()
            return False

        print("  [クリップ %d回目] 細かく合わせる（高さ %dcm）" % (attempt, S.CLIP_ALT_CM))
        if c.clip(qr, S.CLIP_ALT_CM, min(FINE_TIMEOUT_S, left_s - LAND_RESERVE_S)):
            show(c, "  クリップ成功")
            c.land(xy=c.marker_xy.get(qr, (tx, ty)))
            return True
        print("    精密クリップ失敗 → やり直し")

    c.land()
    return False


def run(argv=None):
    args = build_parser().parse_args(argv)
    if any(l <= 0 for l in args.legs):
        print("✗ --legs は正の数で指定してください")
        return 1
    tx, ty = float(sum(args.legs)), 0.0
    if args.dry:
        api = FakeApi([{"qr": args.qr, "x": tx + args.error, "y": 15.0}], seed=args.seed, yaw0=args.yaw0, abs_frame=args.abs_frame)
    else:
        api = connect_drone()
    c = HulaController(api_instance=api)
    patch_controller_api(c)
    c._patched_by_clip8m = True
    install_stop_handlers()
    c.search_moves = []
    ok = False
    try:
        print("接続OK  電池 %d%%  目標: %.0fcm 先のマーカー %d" % (c.battery(), tx, args.qr))
        start = c.now()
        deadline = start + ATTEMPT_TIME_S
        c.takeoff()
        show(c, "離陸OK")
        advance(c, args.legs)
        show(c, "到着")
        ok = clip_and_land(c, args.qr, tx, ty, deadline)
        print("着陸完了 (%s) 所要時間 %.0f秒" % ("成功" if ok else "未達成", c.now() - start))
        return 0 if ok else 2
    except KeyboardInterrupt:
        print("\nCtrl+C → 着陸します")
        return 1
    except DroneError as e:
        print("✗ エラー:", e)
        return 1
    finally:
        c.emergency_land()
        c.close()


if __name__ == "__main__":
    raise SystemExit(run())