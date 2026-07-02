# ============================================================
# XAU/USD SIGNAL BOT — Bot tín hiệu vàng đa khung thời gian
# ============================================================
# Chạy được ở 2 nơi:
#  - Google Colab (thủ công): điền trực tiếp 3 dòng CONFIG bên dưới
#  - GitHub Actions (tự động, định kỳ): để nguyên CONFIG, khai báo
#    3 giá trị qua Secrets (xem hướng dẫn kèm theo)
# ============================================================

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ============================================================
# CONFIG — ĐIỀN THÔNG TIN CỦA BẠN VÀO ĐÂY (nếu chạy trên Colab)
# ============================================================
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "DÁN_API_KEY_TWELVEDATA_VÀO_ĐÂY")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "DÁN_TOKEN_BOT_TELEGRAM_VÀO_ĐÂY")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "DÁN_CHAT_ID_CỦA_BẠN_VÀO_ĐÂY")

SYMBOL = "XAU/USD"
RISK_PER_TRADE_PIPS = 200   # khoảng cách SL mặc định (điểm), có thể chỉnh
SIGNAL_THRESHOLD = 5        # chỉ gửi Telegram khi |điểm tổng hợp| >= giá trị này (đã tối ưu qua backtest sau khi thêm OB+Inside Bar: ngưỡng=5, TP:SL=2.0 cho kỳ vọng dương tốt trên mẫu 48 lệnh)

ADX_MIN = 20                # ADX dưới mức này coi là thị trường đi ngang -> không khuyến nghị vào lệnh
SESSION_FILTER_ENABLED = True   # bật/tắt bộ lọc phiên thanh khoản cao
SESSION_START_UTC = 7       # 07:00 UTC ~ 14:00 giờ VN (mở phiên London)
SESSION_END_UTC = 21        # 21:00 UTC ~ 04:00 giờ VN hôm sau (đóng phiên New York)

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")  # để trống nếu không dùng cảnh báo tin tức
NEWS_WARNING_MINUTES = 45   # cảnh báo nếu có tin quan trọng trong vòng X phút tới

SIGNAL_LOG_PATH = "signal_log.json"   # file lưu lịch sử tín hiệu để tự tính tỷ lệ thắng/thua
SIGNAL_LOG_MAX = 300                  # số bản ghi tối đa giữ lại trong file log
SIGNAL_TIMEOUT_HOURS = 4              # sau X giờ chưa chạm TP/SL thì coi là hết hạn, không tính thắng/thua

# ============================================================
# 1. LẤY DỮ LIỆU GIÁ TỪ TWELVE DATA
# ============================================================
def get_ohlc(interval, outputsize=100):
    """
    interval: '5min', '15min', '30min', '1h'
    Trả về DataFrame với cột: datetime, open, high, low, close
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if "values" not in data:
        raise Exception(f"Lỗi lấy dữ liệu ({interval}): {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def resample_ohlc(df_m5, rule):
    """
    Gộp nến M5 thành khung lớn hơn (15min/30min/1h) NGAY TRONG MÁY,
    không cần gọi thêm API -> tiết kiệm request, cho phép chạy nhanh hơn.
    rule: '15min', '30min', '1h'
    """
    df = df_m5.set_index("datetime")
    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    return out.reset_index()


# ============================================================
# 2. CHỈ BÁO KỸ THUẬT
# ============================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def detect_trend(df, fast=9, slow=21):
    """Trả về 'up', 'down' dựa trên EMA nhanh vs EMA chậm"""
    df = df.copy()
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    last = df.iloc[-1]
    return "up" if last["ema_fast"] > last["ema_slow"] else "down"


def detect_candle_pattern(df):
    """Phát hiện Bullish/Bearish Engulfing và Doji trên 2 nến gần nhất"""
    if len(df) < 2:
        return "none"
    prev, curr = df.iloc[-2], df.iloc[-1]

    body_curr = abs(curr["close"] - curr["open"])
    range_curr = curr["high"] - curr["low"]

    # Doji: thân nến rất nhỏ so với biên độ
    if range_curr > 0 and body_curr / range_curr < 0.1:
        return "doji"

    # Bullish Engulfing: nến hiện tại xanh, "nuốt" thân nến đỏ trước đó
    if (prev["close"] < prev["open"] and curr["close"] > curr["open"]
            and curr["close"] >= prev["open"] and curr["open"] <= prev["close"]):
        return "bullish_engulfing"

    # Bearish Engulfing
    if (prev["close"] > prev["open"] and curr["close"] < curr["open"]
            and curr["open"] >= prev["close"] and curr["close"] <= prev["open"]):
        return "bearish_engulfing"

    return "none"


def detect_bos(df, lookback=20):
    """
    Break of Structure đơn giản: giá hiện tại có phá đỉnh/đáy gần nhất không.
    Trả về 'up' (phá đỉnh), 'down' (phá đáy), hoặc None.
    """
    recent = df.iloc[-lookback:-1]
    curr_close = df.iloc[-1]["close"]
    if curr_close > recent["high"].max():
        return "up"
    if curr_close < recent["low"].min():
        return "down"
    return None


def detect_order_block(df, lookback=20):
    """
    Order Block đơn giản (không phải chuẩn SMC chính thức):
    tìm nến cuối cùng đi ngược hướng trước một đợt di chuyển mạnh.
    - Nến giảm cuối cùng trước đợt tăng mạnh -> Order Block "bullish"
    - Nến tăng cuối cùng trước đợt giảm mạnh -> Order Block "bearish"
    """
    recent = df.iloc[-lookback:].reset_index(drop=True)
    if len(recent) < 6:
        return None

    avg_body = (recent["close"] - recent["open"]).abs().mean()
    if avg_body == 0:
        return None

    for i in range(len(recent) - 4, 0, -1):
        candle = recent.iloc[i]
        next3 = recent.iloc[i + 1:i + 4]
        if len(next3) < 3:
            continue
        body = abs(candle["close"] - candle["open"])
        is_down = candle["close"] < candle["open"]
        is_up = candle["close"] > candle["open"]
        move_up = next3["close"].iloc[-1] - candle["close"]
        move_down = candle["close"] - next3["close"].iloc[-1]

        if is_down and move_up > avg_body * 2 and body > avg_body * 0.5:
            return {"type": "bullish", "zone": (candle["low"], candle["high"])}
        if is_up and move_down > avg_body * 2 and body > avg_body * 0.5:
            return {"type": "bearish", "zone": (candle["low"], candle["high"])}
    return None


def detect_inside_bar_setup(df, atr_series, mother_min_atr_mult=1.5, inside_max_ratio=0.6, max_inside_bars=6):
    """
    Mẫu hình "nến mẹ - nến con" (mother bar / inside bar) theo price action:
    - Nến MẸ: biên độ (high-low) >= 1.5x ATR -> nến biến động mạnh, "quyết định" rõ ràng
    - Nến CON: 1 hoặc nhiều nến liên tiếp nằm GỌN bên trong biên độ nến mẹ -> vùng tích lũy/do dự
    - Cụm nến con phải co lại đủ nhỏ (<=60% biên độ nến mẹ) mới coi là setup "đẹp"
    - Breakout CHỈ được xác nhận khi có nến ĐÓNG CỬA vượt hẳn qua đỉnh/đáy nến mẹ
      (chỉ chạm/chọc râu qua không tính -> lọc bớt false breakout)

    Quét lùi từ nến gần nhất để tìm setup đang hoạt động. Trả về None nếu không tìm thấy.
    """
    n = len(df)
    if n < max_inside_bars + 2:
        return None

    current = df.iloc[-1]  # nến gần nhất - ứng viên breakout hoặc vẫn đang là nến con

    for mother_offset in range(2, max_inside_bars + 2):
        mother_idx = n - 1 - mother_offset
        if mother_idx < 0:
            break
        mother = df.iloc[mother_idx]
        mother_range = mother["high"] - mother["low"]
        atr_at_mother = atr_series.iloc[mother_idx]
        if pd.isna(atr_at_mother) or atr_at_mother <= 0:
            continue
        if mother_range < mother_min_atr_mult * atr_at_mother:
            continue  # nến này không đủ "dài" để làm nến mẹ

        inside_bars = df.iloc[mother_idx + 1: n - 1]
        if len(inside_bars) == 0:
            continue

        all_inside = (inside_bars["high"] <= mother["high"]).all() and (inside_bars["low"] >= mother["low"]).all()
        if not all_inside:
            continue

        cluster_range = inside_bars["high"].max() - inside_bars["low"].min()
        if cluster_range > mother_range * inside_max_ratio:
            continue  # nến con chưa co lại đủ chặt, setup chưa "đẹp"

        touched_high = bool((inside_bars["high"] >= mother["high"] - 0.05 * mother_range).any())
        touched_low = bool((inside_bars["low"] <= mother["low"] + 0.05 * mother_range).any())

        breakout = None
        if current["close"] > mother["high"]:
            breakout = "up"
        elif current["close"] < mother["low"]:
            breakout = "down"

        return {
            "mother_high": mother["high"],
            "mother_low": mother["low"],
            "num_inside_bars": len(inside_bars),
            "touched_high": touched_high,
            "touched_low": touched_low,
            "breakout": breakout,
        }

    return None


def adx(df, period=14):
    """
    ADX (Average Directional Index) — đo ĐỘ MẠNH của xu hướng, không quan tâm hướng.
    ADX < 20: xu hướng yếu / thị trường đi ngang -> tín hiệu trend dễ sai.
    ADX > 25: xu hướng đang rõ ràng, tín hiệu trend đáng tin hơn.
    """
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_.fillna(0)


def is_active_session(now_utc=None):
    """
    Kiểm tra hiện tại có đang trong phiên thanh khoản cao (London + New York) không.
    Ngoài khung này, giá dễ đi ngang/nhiễu, tín hiệu kém tin cậy hơn.
    """
    if not SESSION_FILTER_ENABLED:
        return True
    now_utc = now_utc or datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_START_UTC <= SESSION_END_UTC:
        return SESSION_START_UTC <= hour < SESSION_END_UTC
    return hour >= SESSION_START_UTC or hour < SESSION_END_UTC


def check_upcoming_news():
    """
    Kiểm tra tin kinh tế quan trọng (USD, high impact) sắp ra trong NEWS_WARNING_MINUTES phút tới.
    Dùng Finnhub (cần FINNHUB_API_KEY, để trống thì bỏ qua tính năng này).
    Trả về tên sự kiện gần nhất nếu có, hoặc None. Lỗi mạng/API sẽ bị bỏ qua êm (không làm chết bot).
    """
    if not FINNHUB_API_KEY:
        return None
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {"from": today, "to": today, "token": FINNHUB_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        events = data.get("economicCalendar", data.get("data", []))
        now_utc = datetime.now(timezone.utc)

        for ev in events:
            impact = str(ev.get("impact", "")).lower()
            country = str(ev.get("country", ev.get("economy", ""))).upper()
            if impact not in ("3", "high") or country not in ("US", "USD"):
                continue
            ev_time_str = ev.get("time") or ev.get("data")
            if not ev_time_str:
                continue
            try:
                ev_time = pd.to_datetime(ev_time_str, utc=True)
            except Exception:
                continue
            minutes_away = (ev_time - now_utc).total_seconds() / 60
            if 0 <= minutes_away <= NEWS_WARNING_MINUTES:
                return f"{ev.get('event', ev.get('name', 'Tin quan trọng'))} lúc {ev_time.strftime('%H:%M UTC')}"
        return None
    except Exception:
        return None  # không để lỗi API tin tức làm chết cả bot


def rsi(series, period=14):
    """RSI chuẩn — đo quá mua (>70) / quá bán (<30)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def atr(df, period=14):
    """Average True Range — đo mức độ biến động hiện tại"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def support_resistance(df, lookback=30):
    """Vùng hỗ trợ/kháng cự gần nhất = đáy/đỉnh gần nhất trong lookback nến"""
    recent = df.iloc[-lookback:]
    return {"support": recent["low"].min(), "resistance": recent["high"].max()}


def fibonacci_levels(df, lookback=50):
    """
    Tính các mức Fibonacci retracement/extension từ đợt sóng (đỉnh-đáy) gần nhất
    trong 'lookback' nến. Dùng để tham khảo vùng SL/TP hợp lý (không thay thế
    cách tính ATR đang dùng, chỉ là lớp xác nhận thêm - confluence check).
    """
    recent = df.iloc[-lookback:].reset_index(drop=True)
    if len(recent) < 10:
        return None

    high_idx = recent["high"].idxmax()
    low_idx = recent["low"].idxmin()
    swing_high = recent.loc[high_idx, "high"]
    swing_low = recent.loc[low_idx, "low"]
    diff = swing_high - swing_low
    if diff <= 0:
        return None

    # Nếu đáy hình thành SAU đỉnh -> sóng đang giảm gần nhất -> retracement tính từ trên xuống
    # Nếu đỉnh hình thành SAU đáy -> sóng đang tăng gần nhất -> retracement tính từ dưới lên
    uptrend_leg = low_idx < high_idx

    if uptrend_leg:
        levels = {
            "0.0": swing_high,
            "23.6": swing_high - diff * 0.236,
            "38.2": swing_high - diff * 0.382,
            "50.0": swing_high - diff * 0.5,
            "61.8": swing_high - diff * 0.618,
            "78.6": swing_high - diff * 0.786,
            "100.0": swing_low,
            "ext_127.2": swing_high + diff * 0.272,
            "ext_161.8": swing_high + diff * 0.618,
        }
    else:
        levels = {
            "0.0": swing_low,
            "23.6": swing_low + diff * 0.236,
            "38.2": swing_low + diff * 0.382,
            "50.0": swing_low + diff * 0.5,
            "61.8": swing_low + diff * 0.618,
            "78.6": swing_low + diff * 0.786,
            "100.0": swing_high,
            "ext_127.2": swing_low - diff * 0.272,
            "ext_161.8": swing_low - diff * 0.618,
        }

    return {"swing_high": swing_high, "swing_low": swing_low, "uptrend_leg": uptrend_leg, "levels": levels}


def fib_confluence_note(fib, entry, sl, tp1, atr_value, tolerance_mult=0.5):
    """
    Kiểm tra SL/TP hiện tại có 'trùng' (nằm gần) 1 mức Fib quan trọng không.
    Nếu trùng -> tăng độ tin cậy, trả về ghi chú mô tả. Ngưỡng 'trùng' = tolerance_mult * ATR.
    """
    if not fib:
        return None
    tolerance = atr_value * tolerance_mult
    notes = []
    key_levels = ["38.2", "50.0", "61.8", "ext_127.2", "ext_161.8"]
    for name in key_levels:
        price = fib["levels"][name]
        if abs(sl - price) <= tolerance:
            notes.append(f"SL gần trùng Fib {name}%")
        if abs(tp1 - price) <= tolerance:
            notes.append(f"TP1 gần trùng Fib {name}%")
    return "; ".join(notes) if notes else None


def detect_fvg(df):
    """
    Fair Value Gap đơn giản: khoảng trống giữa nến[-3] và nến[-1]
    (không giao nhau giữa high nến 1 và low nến 3, hoặc ngược lại)
    """
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if c1["high"] < c3["low"]:
        return {"type": "bullish", "zone": (c1["high"], c3["low"])}
    if c1["low"] > c3["high"]:
        return {"type": "bearish", "zone": (c3["high"], c1["low"])}
    return None


# ============================================================
# 2b. TỰ THEO DÕI KẾT QUẢ TÍN HIỆU (WIN-RATE TRACKING)
# ============================================================
def load_signal_log():
    if not os.path.exists(SIGNAL_LOG_PATH):
        return []
    try:
        with open(SIGNAL_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_signal_log(log):
    log = log[-SIGNAL_LOG_MAX:]
    with open(SIGNAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def update_signal_outcomes(log, current_price):
    """Kiểm tra các tín hiệu 'pending' cũ đã chạm TP1, chạm SL, hay hết hạn chưa."""
    now = datetime.now(timezone.utc)
    for rec in log:
        if rec.get("status") != "pending":
            continue
        try:
            rec_time = datetime.fromisoformat(rec["time_iso"])
        except Exception:
            rec["status"] = "expired"
            continue

        if rec["direction"] == "BUY":
            if current_price >= rec["tp1"]:
                rec["status"] = "win"
            elif current_price <= rec["sl"]:
                rec["status"] = "loss"
        else:  # SELL
            if current_price <= rec["tp1"]:
                rec["status"] = "win"
            elif current_price >= rec["sl"]:
                rec["status"] = "loss"

        if rec["status"] == "pending" and (now - rec_time) > timedelta(hours=SIGNAL_TIMEOUT_HOURS):
            rec["status"] = "expired"
    return log


def append_signal(log, sig):
    """Thêm tín hiệu vừa tạo (nếu có hướng BUY/SELL) vào log để theo dõi sau này."""
    if not sig.get("direction"):
        return log
    log.append({
        "time_iso": datetime.now(timezone.utc).isoformat(),
        "direction": sig["direction"],
        "entry": sig["entry"],
        "sl": sig["sl"],
        "tp1": sig["tp1"],
        "score": sig["score"],
        "mode": sig.get("signal_mode", "trend"),  # "trend" hoặc "mean_reversion" - để đánh giá riêng từng loại
        "status": "pending",
    })
    return log


def compute_win_rate(log, mode=None):
    """
    Tính tỷ lệ thắng/thua. Nếu truyền mode ("trend" hoặc "mean_reversion"),
    chỉ tính riêng loại đó -> cho phép đánh giá độc lập 2 chiến lược khác nhau.
    Các bản ghi log cũ (trước khi có trường 'mode') được coi là 'trend' để không mất dữ liệu.
    """
    if mode:
        log = [r for r in log if r.get("mode", "trend") == mode]
    closed = [r for r in log if r.get("status") in ("win", "loss")]
    wins = [r for r in closed if r["status"] == "win"]
    if not closed:
        return None
    return {
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "total": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
    }


def mean_reversion_signal(current_price, sr, rsi_value, atr_value, near_threshold=0.25):
    """
    Chiến lược RIÊNG cho lúc thị trường sideway (ADX thấp) — khác hẳn logic trend-following.
    Ý tưởng: trong sideway, giá dao động qua lại giữa hỗ trợ/kháng cự thay vì đi theo xu hướng.
    - Giá gần ĐÁY range + RSI thấp (chưa quá bán hẳn nhưng nghiêng yếu) -> kỳ vọng bật lên -> BUY
    - Giá gần ĐỈNH range + RSI cao -> kỳ vọng giảm trở lại -> SELL
    R:R thấp hơn logic trend (range hẹp thì target cũng phải gần, không kỳ vọng xa như lúc có trend).
    """
    support, resistance = sr["support"], sr["resistance"]
    range_size = resistance - support
    if range_size <= 0:
        return None

    position = (current_price - support) / range_size  # 0 = tại đáy, 1 = tại đỉnh range
    mid = (support + resistance) / 2

    direction = None
    if position <= near_threshold and rsi_value <= 45:
        direction = "BUY"
    elif position >= (1 - near_threshold) and rsi_value >= 55:
        direction = "SELL"

    if not direction:
        return None

    buffer = atr_value * 0.5
    if direction == "BUY":
        sl = support - buffer
        tp1 = mid
        tp2 = resistance
        tp3 = resistance + range_size * 0.3
    else:
        sl = resistance + buffer
        tp1 = mid
        tp2 = support
        tp3 = support - range_size * 0.3

    # Bỏ qua nếu target quá gần entry (range quá hẹp, không đáng vào lệnh sau khi trừ phí)
    if abs(tp1 - current_price) < atr_value * 0.5:
        return None

    return {
        "direction": direction, "entry": current_price, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "position_in_range": round(position, 2),
    }


def check_entry_chase(direction, current_price, ob, atr_value, max_atr_distance=2.5):
    """
    Kiểm tra giá hiện tại đã chạy quá xa vùng Order Block chưa (nguy cơ "mua đuổi/bán đuổi").
    Nếu quá xa (>= max_atr_distance x ATR), trả về gợi ý entry CHỜ (limit) tại biên gần
    của OB thay vì entry thị trường ngay - giống nguyên tắc "không mua đuổi, chờ giá về".
    """
    if not ob:
        return None
    zone_low, zone_high = ob["zone"]

    if direction == "BUY":
        ref_edge = zone_high  # biên gần nhất để chờ giá hồi về khi đang ở trên vùng OB
        distance = current_price - ref_edge
    else:
        ref_edge = zone_low
        distance = ref_edge - current_price

    if distance <= 0 or atr_value <= 0:
        return None  # giá còn trong/chưa vượt vùng OB, chưa cần cảnh báo

    distance_atr = distance / atr_value
    if distance_atr >= max_atr_distance:
        return {"distance_atr": round(distance_atr, 1), "suggested_entry": ref_edge}
    return None


# ============================================================
# 3. LOGIC TẠO TÍN HIỆU
# ============================================================
def generate_signal():
    # Chỉ gọi API 1 lần (lấy nhiều nến M5), sau đó tự gộp thành M15/M30/H1
    # -> tiết kiệm request, cho phép chạy mỗi 5 phút mà vẫn trong hạn mức free
    df_m5 = get_ohlc("5min", outputsize=1000)  # ~3.5 ngày dữ liệu M5
    df_m15 = resample_ohlc(df_m5, "15min")
    df_m30 = resample_ohlc(df_m5, "30min")
    df_h1 = resample_ohlc(df_m5, "1h")

    trend_m5 = detect_trend(df_m5)
    trend_m15 = detect_trend(df_m15)
    trend_m30 = detect_trend(df_m30)

    pattern = detect_candle_pattern(df_m5)
    bos = detect_bos(df_m5)
    fvg = detect_fvg(df_m5)
    ob = detect_order_block(df_m5)

    rsi_m5 = rsi(df_m5["close"]).iloc[-1]
    atr_m5 = atr(df_m5).iloc[-1]
    atr_m15_series = atr(df_m15)
    adx_m15 = adx(df_m15).iloc[-1]
    sr = support_resistance(df_m5)
    fib = fibonacci_levels(df_m30, lookback=50)
    current_price = df_m5.iloc[-1]["close"]

    # Mẫu hình nến mẹ - nến con (Inside Bar), quét trên M15 theo đề xuất
    # (M5 quá nhiễu cho pattern này, M15 phản ánh cấu trúc rõ hơn)
    inside_bar = detect_inside_bar_setup(df_m15, atr_m15_series)

    session_ok = is_active_session()
    news_warning = check_upcoming_news()

    # --- Chấm điểm đơn giản (bạn có thể chỉnh trọng số) ---
    # Thang điểm tối đa: trend M5/M15/M30 (±1 mỗi cái) + pattern (±2) + BOS (±1) + OB (±1)
    #                     + Inside Bar breakout (±1) = ±8
    score = 0
    if trend_m5 == "up": score += 1
    if trend_m15 == "up": score += 1
    if trend_m30 == "up": score += 1
    if trend_m5 == "down": score -= 1
    if trend_m15 == "down": score -= 1
    if trend_m30 == "down": score -= 1
    if pattern == "bullish_engulfing": score += 2
    if pattern == "bearish_engulfing": score -= 2
    if bos == "up": score += 1
    if bos == "down": score -= 1
    if ob and ob["type"] == "bullish": score += 1
    if ob and ob["type"] == "bearish": score -= 1
    if inside_bar and inside_bar["breakout"] == "up": score += 1
    if inside_bar and inside_bar["breakout"] == "down": score -= 1

    direction = None
    block_reason = None
    signal_mode = "trend"

    if score >= SIGNAL_THRESHOLD:
        direction = "BUY"
    elif score <= -SIGNAL_THRESHOLD:
        direction = "SELL"

    is_sideway = adx_m15 < ADX_MIN

    # --- Các bộ lọc chặn tín hiệu trend nếu điều kiện thị trường không thuận lợi ---
    if direction:
        if is_sideway:
            direction = None  # tín hiệu trend không đáng tin khi sideway, xử lý riêng bên dưới
        elif news_warning:
            direction = None
            block_reason = f"Sắp có tin quan trọng: {news_warning} -> tạm ẩn khuyến nghị để tránh SL bị quét"

    # --- Nếu đang sideway: thử chiến lược mean-reversion riêng (khác hẳn trend-following) ---
    mr = None
    if is_sideway and not news_warning:
        mr = mean_reversion_signal(current_price, sr, rsi_m5, atr_m5)
        if mr:
            direction = mr["direction"]
            signal_mode = "mean_reversion"
        else:
            block_reason = (f"ADX(M15)={adx_m15:.1f} < {ADX_MIN} -> thị trường đi ngang, "
                             f"và giá chưa ở vùng biên range đủ rõ để đánh mean-reversion")
    elif is_sideway and news_warning:
        block_reason = f"Sắp có tin quan trọng: {news_warning} -> tạm ẩn mọi khuyến nghị"

    liquidity_note = None
    if not session_ok:
        liquidity_note = "Đang ngoài phiên thanh khoản cao (London/New York) — giá dễ nhiễu/đi ngang hơn bình thường, cân nhắc khối lượng nhỏ hơn nếu vào lệnh."

    # % thay đổi so với ~24 giờ trước (ước lượng thô từ khung H1)
    try:
        ref_price = df_h1.iloc[max(0, len(df_h1) - 24)]["close"]
        pct_change = (current_price - ref_price) / ref_price * 100
    except Exception:
        pct_change = None

    # Mức độ mạnh của tín hiệu, quy ra thang 10 để dễ hình dung
    strength_10 = round(min(10, abs(score) / 8 * 10), 1)

    # --- Nhận định tổng quan (ghép các yếu tố thành 1-2 câu dễ hiểu) ---
    notes = []
    if rsi_m5 >= 70:
        notes.append("RSI cho thấy vùng quá mua, cẩn trọng nếu mua đuổi")
    elif rsi_m5 <= 30:
        notes.append("RSI cho thấy vùng quá bán, cẩn trọng nếu bán đuổi")
    else:
        notes.append("RSI trung tính, chưa quá mua/quá bán")

    trend_count_up = sum(1 for t in [trend_m5, trend_m15, trend_m30] if t == "up")
    if trend_count_up == 3:
        notes.append("cả 3 khung đều đồng thuận tăng")
    elif trend_count_up == 0:
        notes.append("cả 3 khung đều đồng thuận giảm")
    else:
        notes.append("các khung thời gian đang lệch hướng nhau, độ tin cậy thấp hơn")

    dist_to_res = sr["resistance"] - current_price
    dist_to_sup = current_price - sr["support"]
    if dist_to_res < dist_to_sup:
        notes.append(f"giá đang gần kháng cự {sr['resistance']:.2f} hơn, khả năng bị cản")
    else:
        notes.append(f"giá đang gần hỗ trợ {sr['support']:.2f} hơn, khả năng được nâng đỡ")

    overview = "; ".join(notes) + "."

    result = {
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "price": current_price,
        "pct_change": pct_change,
        "score": score,
        "strength_10": strength_10,
        "direction": direction,
        "trend_m5": trend_m5,
        "trend_m15": trend_m15,
        "trend_m30": trend_m30,
        "pattern": pattern,
        "bos": bos,
        "fvg": fvg,
        "ob": ob,
        "inside_bar": inside_bar,
        "rsi": rsi_m5,
        "atr": atr_m5,
        "adx": adx_m15,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "overview": overview,
        "session_ok": session_ok,
        "liquidity_note": liquidity_note,
        "news_warning": news_warning,
        "block_reason": block_reason,
        "fib": fib,
        "fib_note": None,
        "signal_mode": signal_mode,
        "entry_type": "market",
        "chase_warning": None,
    }

    if direction and signal_mode == "mean_reversion":
        # Dùng thẳng SL/TP đã tính trong mean_reversion_signal (dựa trên vùng range, không dùng ATR*3
        # vì range hẹp không đủ chỗ cho target xa như lúc có trend)
        result.update({
            "entry": mr["entry"], "sl": mr["sl"],
            "tp1": mr["tp1"], "tp2": mr["tp2"], "tp3": mr["tp3"],
        })
        result["fib_note"] = fib_confluence_note(fib, mr["entry"], mr["sl"], mr["tp1"], atr_m5)

    elif direction:
        # Kiểm tra giá hiện tại đã chạy quá xa vùng OB chưa -> tránh khuyến nghị mua/bán đuổi
        chase = check_entry_chase(direction, current_price, ob, atr_m5)
        entry = chase["suggested_entry"] if chase else current_price
        entry_type = "limit" if chase else "market"

        # Dùng ATR để đặt SL theo biến động thực tế của thị trường (thay vì số pip cố định cứng nhắc)
        sl_distance = max(atr_m5 * 1.5, RISK_PER_TRADE_PIPS * 0.01 * 0.5)
        # Tỷ lệ TP:SL = 2.0 -> áp dụng từ kết quả backtest (kỳ vọng dương nhất trên mẫu đủ lớn,
        # xem run_sweep() trong backtest.py). TP2/TP3 đặt xa hơn TP1 để chốt lời từng phần.
        tp1_distance = sl_distance * 2.0

        if direction == "BUY":
            sl = entry - sl_distance
            tp1 = entry + tp1_distance
            tp2 = entry + tp1_distance * 1.3
            tp3 = entry + tp1_distance * 1.6
        else:
            sl = entry + sl_distance
            tp1 = entry - tp1_distance
            tp2 = entry - tp1_distance * 1.3
            tp3 = entry - tp1_distance * 1.6

        result.update({
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "entry_type": entry_type, "chase_warning": chase,
        })
        result["fib_note"] = fib_confluence_note(fib, entry, sl, tp1, atr_m5)

    return result


# ============================================================
# 4. FORMAT TIN NHẮN & GỬI TELEGRAM
# ============================================================
def format_message(sig, win_stats=None):
    trend_icon = lambda t: "⬆️" if t == "up" else "⬇️"
    trend_label = lambda t: "Tăng" if t == "up" else "Giảm"

    icon = "🟢" if sig["direction"] == "BUY" else ("🔴" if sig["direction"] == "SELL" else "⚪")

    lines = []
    lines.append(f"⚡ SCALP XAU/USD   {sig['time']}")
    price_line = f"💰 {sig['price']:.2f}"
    if sig["pct_change"] is not None:
        price_line += f"   ({sig['pct_change']:+.2f}% /24h)"
    lines.append(price_line)
    lines.append(f"📶 Độ mạnh tín hiệu: {sig['strength_10']}/10")
    lines.append("─────────────────────")

    lines.append("📊 Xu hướng đa khung:")
    lines.append(f"   M5:{trend_icon(sig['trend_m5'])} {trend_label(sig['trend_m5'])}   "
                  f"M15:{trend_icon(sig['trend_m15'])} {trend_label(sig['trend_m15'])}   "
                  f"M30:{trend_icon(sig['trend_m30'])} {trend_label(sig['trend_m30'])}")

    if sig["ob"]:
        z = sig["ob"]["zone"]
        lines.append(f"🟦 Order Block ({sig['ob']['type']}): {z[0]:.2f}–{z[1]:.2f}")
    if sig["fvg"]:
        z = sig["fvg"]["zone"]
        lines.append(f"📊 FVG ({sig['fvg']['type']}): {z[0]:.2f}–{z[1]:.2f}")
    if sig["bos"]:
        lines.append(f"🔀 BOS: vừa phá {'đỉnh' if sig['bos']=='up' else 'đáy'} gần nhất (M5)")
    if sig["pattern"] != "none":
        lines.append(f"🕯️ Mẫu nến M5: {sig['pattern']}")
    if sig.get("inside_bar"):
        ib = sig["inside_bar"]
        rej = []
        if ib["touched_high"]: rej.append("chạm đỉnh")
        if ib["touched_low"]: rej.append("chạm đáy")
        rej_txt = f", đã {'/'.join(rej)}" if rej else ""
        bo_txt = {"up": "ĐÃ BREAKOUT LÊN ✅", "down": "ĐÃ BREAKOUT XUỐNG ✅", None: "chưa breakout, đang chờ"}[ib["breakout"]]
        lines.append(f"📦 Inside Bar (M15): {ib['num_inside_bars']} nến con trong "
                      f"{ib['mother_low']:.2f}–{ib['mother_high']:.2f}{rej_txt} — {bo_txt}")

    lines.append(f"📈 RSI(14): {sig['rsi']:.1f}   |   ATR(14): {sig['atr']:.2f}   |   ADX(M15): {sig['adx']:.1f}")
    lines.append(f"🧱 Hỗ trợ: {sig['support']:.2f}   |   Kháng cự: {sig['resistance']:.2f}")
    if sig.get("fib"):
        lv = sig["fib"]["levels"]
        lines.append(f"🔢 Fib (M30, {'sóng tăng' if sig['fib']['uptrend_leg'] else 'sóng giảm'}): "
                      f"38.2%={lv['38.2']:.2f}  50%={lv['50.0']:.2f}  61.8%={lv['61.8']:.2f}  "
                      f"| Ext 161.8%={lv['ext_161.8']:.2f}")
    if sig.get("fib_note"):
        lines.append(f"✨ Đồng thuận Fib: {sig['fib_note']}")
    lines.append(f"🕐 Phiên thanh khoản cao: {'Có' if sig['session_ok'] else 'Không'}")
    if sig["liquidity_note"]:
        lines.append(f"⚠️ {sig['liquidity_note']}")
    if sig["news_warning"]:
        lines.append(f"📰 Cảnh báo tin: {sig['news_warning']}")

    lines.append("─────────────────────")
    lines.append(f"📐 Điểm tổng hợp: {sig['score']} / ±8")
    lines.append(f"🧠 Nhận định: {sig['overview']}")

    if sig["direction"]:
        lines.append("")
        if sig.get("signal_mode") == "mean_reversion":
            lines.append(f"{icon} {sig['direction']}   🔁 MEAN-REVERSION (thị trường sideway)")
            lines.append("   ⚠️ Chiến lược đánh biên range, KHÁC với trend-following thông thường —")
            lines.append("   target gần hơn, phù hợp lúc thị trường không có xu hướng rõ.")
        else:
            lines.append(f"{icon} {sig['direction']}")

        if sig.get("chase_warning"):
            cw = sig["chase_warning"]
            lines.append(f"   ⏳ Giá đã chạy {cw['distance_atr']}x ATR khỏi vùng OB — CHỜ GIÁ VỀ, không mua/bán đuổi")
            lines.append(f"📍 Entry (chờ, limit): {sig['entry']:.2f}")
        else:
            lines.append(f"📍 Entry (vào ngay, market): {sig['entry']:.2f}")

        lines.append(f"🛑 SL: {sig['sl']:.2f}")
        lines.append(f"✅ TP: {sig['tp1']:.2f} / {sig['tp2']:.2f} / {sig['tp3']:.2f}")
    elif sig["block_reason"]:
        lines.append("")
        lines.append(f"⚪ {sig['block_reason']}")
    else:
        lines.append("")
        lines.append("⚪ Chưa đủ tín hiệu rõ ràng để vào lệnh lúc này")

    if win_stats:
        lines.append("─────────────────────")
        lines.append("🎯 Lịch sử tín hiệu (đánh giá riêng từng chiến lược):")
        trend_stats = win_stats.get("trend")
        mr_stats = win_stats.get("mean_reversion")
        if trend_stats:
            lines.append(f"   📈 Trend: {trend_stats['wins']} thắng / {trend_stats['losses']} thua "
                          f"({trend_stats['win_rate']}% trên {trend_stats['total']} lệnh)")
        else:
            lines.append("   📈 Trend: chưa đủ dữ liệu")
        if mr_stats:
            lines.append(f"   🔁 Mean-Reversion: {mr_stats['wins']} thắng / {mr_stats['losses']} thua "
                          f"({mr_stats['win_rate']}% trên {mr_stats['total']} lệnh)")
        else:
            lines.append("   🔁 Mean-Reversion: chưa đủ dữ liệu")

    lines.append("")
    lines.append("⚠️ Chỉ tham khảo | Quản lý vốn 1-2%")

    return "\n".join(lines)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    r = requests.post(url, data=payload, timeout=15)
    if r.status_code != 200:
        raise Exception(f"Lỗi gửi Telegram: {r.text}")
    return r.json()


# ============================================================
# 5. CHẠY BOT
# ============================================================
if __name__ == "__main__":
    print("Đang lấy dữ liệu và phân tích...")
    signal = generate_signal()

    # --- Cập nhật kết quả các tín hiệu cũ, thêm tín hiệu mới, tính tỷ lệ thắng/thua RIÊNG từng loại ---
    log = load_signal_log()
    log = update_signal_outcomes(log, signal["price"])
    log = append_signal(log, signal)
    save_signal_log(log)
    win_stats = {
        "trend": compute_win_rate(log, mode="trend"),
        "mean_reversion": compute_win_rate(log, mode="mean_reversion"),
    }

    message = format_message(signal, win_stats=win_stats)
    print(message)

    print("\nĐang gửi vào Telegram...")
    send_telegram(message)
    print("Đã gửi xong! Kiểm tra Telegram của bạn.")
