# ============================================================
# XAU/USD SIGNAL BOT — Bot tín hiệu vàng đa khung thời gian
# ============================================================
# Chạy được ở 2 nơi:
#  - Google Colab (thủ công): điền trực tiếp 3 dòng CONFIG bên dưới
#  - GitHub Actions (tự động, định kỳ): để nguyên CONFIG, khai báo
#    3 giá trị qua Secrets (xem hướng dẫn kèm theo)
# ============================================================

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIG — ĐIỀN THÔNG TIN CỦA BẠN VÀO ĐÂY (nếu chạy trên Colab)
# ============================================================
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "DÁN_API_KEY_TWELVEDATA_VÀO_ĐÂY")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "DÁN_TOKEN_BOT_TELEGRAM_VÀO_ĐÂY")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "DÁN_CHAT_ID_CỦA_BẠN_VÀO_ĐÂY")

SYMBOL = "XAU/USD"
RISK_PER_TRADE_PIPS = 200   # khoảng cách SL mặc định (điểm), có thể chỉnh
SIGNAL_THRESHOLD = 4        # chỉ gửi Telegram khi |điểm tổng hợp| >= giá trị này (thang điểm tối đa hiện tại là ±7)

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
# 3. LOGIC TẠO TÍN HIỆU
# ============================================================
def generate_signal():
    df_m5 = get_ohlc("5min")
    df_m15 = get_ohlc("15min")
    df_m30 = get_ohlc("30min")
    df_h1 = get_ohlc("1h")

    trend_m5 = detect_trend(df_m5)
    trend_m15 = detect_trend(df_m15)
    trend_m30 = detect_trend(df_m30)

    pattern = detect_candle_pattern(df_m5)
    bos = detect_bos(df_m5)
    fvg = detect_fvg(df_m5)
    ob = detect_order_block(df_m5)

    rsi_m5 = rsi(df_m5["close"]).iloc[-1]
    atr_m5 = atr(df_m5).iloc[-1]
    sr = support_resistance(df_m5)

    # --- Chấm điểm đơn giản (bạn có thể chỉnh trọng số) ---
    # Thang điểm tối đa: trend M5/M15/M30 (±1 mỗi cái) + pattern (±2) + BOS (±1) + OB (±1) = ±7
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

    direction = None
    if score >= 3:
        direction = "BUY"
    elif score <= -3:
        direction = "SELL"

    current_price = df_m5.iloc[-1]["close"]

    # % thay đổi so với ~24 giờ trước (ước lượng thô từ khung H1)
    try:
        ref_price = df_h1.iloc[max(0, len(df_h1) - 24)]["close"]
        pct_change = (current_price - ref_price) / ref_price * 100
    except Exception:
        pct_change = None

    # Mức độ mạnh của tín hiệu, quy ra thang 10 để dễ hình dung
    strength_10 = round(min(10, abs(score) / 7 * 10), 1)

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
        "rsi": rsi_m5,
        "atr": atr_m5,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "overview": overview,
    }

    if direction:
        entry = current_price
        # Dùng ATR để đặt SL theo biến động thực tế của thị trường (thay vì số pip cố định cứng nhắc)
        sl_distance = max(atr_m5 * 1.5, RISK_PER_TRADE_PIPS * 0.01 * 0.5)

        if direction == "BUY":
            sl = entry - sl_distance
            tp1 = entry + sl_distance
            tp2 = entry + sl_distance * 1.5
            tp3 = entry + sl_distance * 2.5
        else:
            sl = entry + sl_distance
            tp1 = entry - sl_distance
            tp2 = entry - sl_distance * 1.5
            tp3 = entry - sl_distance * 2.5

        result.update({"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3})

    return result


# ============================================================
# 4. FORMAT TIN NHẮN & GỬI TELEGRAM
# ============================================================
def format_message(sig):
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

    lines.append(f"📈 RSI(14): {sig['rsi']:.1f}   |   ATR(14): {sig['atr']:.2f}")
    lines.append(f"🧱 Hỗ trợ: {sig['support']:.2f}   |   Kháng cự: {sig['resistance']:.2f}")

    lines.append("─────────────────────")
    lines.append(f"📐 Điểm tổng hợp: {sig['score']} / ±7")
    lines.append(f"🧠 Nhận định: {sig['overview']}")

    if sig["direction"]:
        lines.append("")
        lines.append(f"{icon} {sig['direction']}")
        lines.append(f"📍 Entry: {sig['entry']:.2f}")
        lines.append(f"🛑 SL: {sig['sl']:.2f}")
        lines.append(f"✅ TP: {sig['tp1']:.2f} / {sig['tp2']:.2f} / {sig['tp3']:.2f}")
    else:
        lines.append("")
        lines.append("⚪ Chưa đủ tín hiệu rõ ràng để vào lệnh lúc này")

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
    message = format_message(signal)
    print(message)

    print("\nĐang gửi vào Telegram...")
    send_telegram(message)
    print("Đã gửi xong! Kiểm tra Telegram của bạn.")
