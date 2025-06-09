# --- Imports ---
import requests
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler, MessageHandler, Filters
from functools import lru_cache
import concurrent.futures
from datetime import datetime, timedelta

# --- Constants ---
TELEGRAM_BOT_TOKEN = "7575210959:AAFzqsnt7Gx3qMBxl4M9gzOFNsAFSKn2W2Q"
TELEGRAM_CHAT_ID = "906015851"
MAX_WORKERS = 12
session = requests.Session()
user_filters = {}

# --- Indicator & Utility Functions ---
@lru_cache(maxsize=1)
def get_usdt_symbols(market="spot"):
    url = {
        "spot": "https://api.binance.com/api/v3/exchangeInfo",
        "futures": "https://fapi.binance.com/fapi/v1/exchangeInfo"
    }[market]
    info = session.get(url).json()
    return [s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and not any(x in s["symbol"] for x in ["UP", "DOWN", "BULL", "BEAR"])]

def fetch_klines(symbol, interval, market):
    base_url = "https://api.binance.com" if market == "spot" else "https://fapi.binance.com"
    endpoint = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    url = f"{base_url}{endpoint}"
    return session.get(url, params={"symbol": symbol, "interval": interval, "limit": 200}).json()


def calculate_macd(close_prices, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) and Signal Line.

    Parameters:
    - close_prices: List of closing prices (list or pandas Series).
    - short_period: The period for the short EMA (default is 12).
    - long_period: The period for the long EMA (default is 26).
    - signal_period: The period for the signal line EMA (default is 9).

    Returns:
    - MACD value (float)
    - Signal line value (float)
    """
    # Calculate short and long EMAs using the given periods
    short_ema = pd.Series(close_prices).ewm(span=short_period).mean()
    long_ema = pd.Series(close_prices).ewm(span=long_period).mean()

    # Calculate MACD
    macd = short_ema - long_ema

    # Calculate signal line (EMA of MACD)
    signal = macd.ewm(span=signal_period).mean()

    # Return the latest values of MACD and Signal Line
    return macd.iloc[-1], signal.iloc[-1]


def compute_rsi(closes, period=14):
    series = pd.Series(closes)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_adx(highs, lows, closes, period=14):
    df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})
    df["TR"] = df["High"].combine(df["Close"].shift(), max) - df["Low"].combine(df["Close"].shift(), min)
    df["+DM"] = df["High"].diff()
    df["-DM"] = -df["Low"].diff()
    df["+DM"] = df["+DM"].where((df["+DM"] > df["-DM"]) & (df["+DM"] > 0), 0.0)
    df["-DM"] = df["-DM"].where((df["-DM"] > df["+DM"]) & (df["-DM"] > 0), 0.0)
    tr = df["TR"].rolling(window=period).sum()
    plus_di = 100 * (df["+DM"].rolling(window=period).sum() / tr)
    minus_di = 100 * (df["-DM"].rolling(window=period).sum() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=period).mean().iloc[-1]

def calculate_rma(tr, period):
    rma = [tr.iloc[0]]  # start with first TR
    alpha = 1 / period
    for i in range(1, len(tr)):
        rma.append((1 - alpha) * rma[-1] + alpha * tr.iloc[i])
    return pd.Series(rma, index=tr.index)


def calculate_atr(highs, lows, closes, period=14):
    df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift())
    df["L-PC"] = abs(df["Low"] - df["Close"].shift())
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    atr = calculate_rma(tr, period)
    return atr.iloc[-1]



def analyze_multi_tf(symbol, market):
    tf_map = [("15m", "15m"), ("1h", "1h"), ("4h", "4h"), ("1d", "1d")]
    rsi_info, macd_info = [], []
    for tf_label, tf_api in tf_map:
        klines = fetch_klines(symbol, tf_api, market)
        closes = [float(k[4]) for k in klines]
        if len(closes) < 26:
            rsi_info.append((tf_label, None, "⚪"))
            macd_info.append((tf_label, "⚪"))
            continue
        rsi_val = compute_rsi(closes)
        rsi_dot = "🟢" if rsi_val >= 52 else "🔴" if rsi_val <= 48 else "⚪"
        macd, signal = calculate_macd(closes)
        macd_dot = "🟢" if macd > signal else "🔴"
        rsi_info.append((tf_label, round(rsi_val, 1), rsi_dot))
        macd_info.append((tf_label, macd_dot))
    return rsi_info, macd_info

def get_btc_header():
    TZ_OFFSET = +2  # Adjust to your local timezone
    market = "spot"
    symbol = "BTCUSDT"
    tf_list = [("5m", 2), ("15m", 3), ("1h", 3), ("4h", 2), ("1d", 2)]

    now_utc = datetime.utcnow()
    local_time = now_utc + timedelta(hours=TZ_OFFSET)
    now = local_time.strftime('%H:%M')

    lines = [f"*🧠 BTCUSDT Summary ({market.title()} {now} CET)*"]
    atr_row = "*ATR Values:*"
    rsi6_row = "*RSI-6 =*"

    atr_period_map = {"5m": 14, "15m": 14, "1h": 10, "4h": 7, "1d": 7}

    for tf_label, N in tf_list:
        try:
            klines = fetch_klines(symbol, tf_label, market)
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]

            if len(closes) < 200:
                lines.append(f"{tf_label}: Not enough data.")
                atr_row += f" `{tf_label}=NA` 🔴 |"
                if tf_label in ["5m", "15m", "1h"]:
                    rsi6_row += f" {tf_label}=NA ⚪ |"
                continue

            # === RSI & MACD ===
            rsi_val = compute_rsi(closes)
            rsi_dot = "🟢" if rsi_val >= 52 else "🔴" if rsi_val <= 48 else "⚪"
            macd_val, signal_val = calculate_macd(closes)
            macd_dot = "🟢" if macd_val > signal_val else "🔴"

            # === SMA7 / SMA25 Cross ===
            sma7_series = pd.Series(closes).rolling(window=7).mean()
            sma25_series = pd.Series(closes).rolling(window=25).mean()
            cross_7_25 = "⚪"
            for i in range(1, N + 1):
                sma7_now = sma7_series.iloc[-i]
                sma25_now = sma25_series.iloc[-i]
                sma7_prev = sma7_series.iloc[-(i + 1)]
                sma25_prev = sma25_series.iloc[-(i + 1)]
                if sma7_now > sma25_now and sma7_prev <= sma25_prev:
                    cross_7_25 = f"🟢 ({i}c)"
                    break
                elif sma7_now < sma25_now and sma7_prev >= sma25_prev:
                    cross_7_25 = f"🔴 ({i}c)"
                    break

            # === SMA50 / SMA200 Cross ===
            sma50_series = pd.Series(closes).rolling(window=50).mean()
            sma200_series = pd.Series(closes).rolling(window=200).mean()
            cross_50_200 = "⚪"
            for i in range(1, N + 1):
                sma50_now = sma50_series.iloc[-i]
                sma200_now = sma200_series.iloc[-i]
                sma50_prev = sma50_series.iloc[-(i + 1)]
                sma200_prev = sma200_series.iloc[-(i + 1)]
                if sma50_now > sma200_now and sma50_prev <= sma200_prev:
                    cross_50_200 = f"🟢 ({i}c)"
                    break
                elif sma50_now < sma200_now and sma50_prev >= sma200_prev:
                    cross_50_200 = f"🔴 ({i}c)"
                    break

            # === ATR ===
            period = atr_period_map.get(tf_label, 14)
            atr = calculate_atr(highs, lows, closes, period=period)
            atr_ratio = atr / closes[-1]

            if closes[-1] >= 100:
                atr_dot = "🟢" if 0.0025 <= atr_ratio <= 0.015 else "🔴"
            elif closes[-1] >= 1:
                atr_dot = "🟢" if 0.01 <= atr_ratio <= 0.03 else "🔴"
            else:
                atr_dot = "🟢" if 0.02 <= atr_ratio <= 0.08 else "🔴"

            atr_row += f" `{tf_label}={atr:.2f}` {atr_dot} |"

            # === RSI-6 (Scalping Insight) ===
            if tf_label in ["5m", "15m", "1h"]:
                rsi6 = compute_rsi(closes, period=6)
                rsi6_dot = "🟢" if rsi6 <= 30 else "🔴" if rsi6 >= 70 else "⚪"
                rsi6_row += f" {tf_label}={rsi6:.1f} {rsi6_dot} |"

            # === Final Summary Line ===
            lines.append(
                f"{tf_label} 🕒 | "
                f"RSI: {rsi_val:.1f} {rsi_dot} | "
                f"MACD: {macd_dot} | "
                f"7x25: {cross_7_25} | "
                f"50x200: {cross_50_200}"
            )

        except Exception as e:
            lines.append(f"{tf_label}: Error - {e}")
            atr_row += f" `{tf_label}=Err` 🔴 |"
            if tf_label in ["5m", "15m", "1h"]:
                rsi6_row += f" {tf_label}=Err ⚪ |"

    lines.append("\n" + atr_row.rstrip("|"))
    lines.append(rsi6_row.rstrip("|"))
    return "\n".join(lines) + "\n\n"




# --- Symbol Filtering Logic ---
# --- Symbol Filtering Logic ---
def process_symbol(symbol, interval, market, filter_type, trend, sma_min=None, sma_max=None, sma_choice=None):
    try:
        # === Fetch Kline Data ===
        klines = fetch_klines(symbol, interval, market)
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        if len(closes) < 60:  # Ensure enough candles
            return None

        # === Select SMA Periods Based on Interval ===
        if interval in ["5m", "15m", "1h"]:
            sma_fast_p, sma_slow_p, sma_third_p, sma_extra = 5, 10, 30, 60
        else:
            sma_fast_p, sma_slow_p, sma_third_p, sma_extra = 7, 25, 99, 200

        # === Calculate SMA values ===
        sma_fast = pd.Series(closes).rolling(window=sma_fast_p).mean().iloc[-1]
        sma_slow = pd.Series(closes).rolling(window=sma_slow_p).mean().iloc[-1]
        sma_third = pd.Series(closes).rolling(window=sma_third_p).mean().iloc[-1]
        sma_extra_val = pd.Series(closes).rolling(window=sma_extra).mean().iloc[-1]

        sma_fast_prev = pd.Series(closes).rolling(window=sma_fast_p).mean().iloc[-2]
        sma_slow_prev = pd.Series(closes).rolling(window=sma_slow_p).mean().iloc[-2]
        sma_third_prev = pd.Series(closes).rolling(window=sma_third_p).mean().iloc[-2]

        price = closes[-1]
        sma_value = {
            'sma5': sma_fast, 'sma10': sma_slow, 'sma30': sma_third, 'sma60': sma_extra_val,
            'sma7': sma_fast, 'sma25': sma_slow, 'sma99': sma_third, 'sma200': sma_extra_val
        }.get(sma_choice, sma_slow)

        sma_pct = (price - sma_value) / sma_value * 100

        # === RSI and MACD ===
        rsi = compute_rsi(closes)
        rsi_6 = compute_rsi(closes, period=6)
        macd_val, signal_val = calculate_macd(closes)
        macd_state = "🟢 Bullish" if macd_val > signal_val else "🔴 Bearish"
        cluster_label = "None"  # Default in case no cluster logic applies


        # === Filter Logic ===
        if filter_type == "sma7x25x99":
            if trend == "bullish" and not (
                sma_fast > sma_slow and sma_fast > sma_third and sma_fast_prev <= sma_third_prev and macd_val > signal_val
            ): return None
            if trend == "bearish" and not (
                sma_fast < sma_slow and sma_fast < sma_third and sma_fast_prev >= sma_third_prev and macd_val < signal_val
            ): return None

        elif filter_type == "smapercent":
            if sma_min is None or sma_max is None:
                return None
            if trend == "bullish" and not (sma_min <= sma_pct <= sma_max and rsi_6 >= 40):
                return None
            if trend == "bearish" and not (-sma_max <= sma_pct <= -sma_min and rsi_6 <= 30):
                return None

        elif filter_type == "smacrossover":
            if trend == "bullish" and not (sma_fast > sma_slow and sma_fast_prev <= sma_slow_prev and macd_val > signal_val):
                return None
            if trend == "bearish" and not (sma_fast < sma_slow and sma_fast_prev >= sma_slow_prev and macd_val < signal_val):
                return None

        elif filter_type == "smacluster":
            sma7 = pd.Series(closes).rolling(window=7).mean().iloc[-1]
            sma25 = pd.Series(closes).rolling(window=25).mean().iloc[-1]
            sma99 = pd.Series(closes).rolling(window=99).mean().iloc[-1]

            sma_max = max(sma7, sma25, sma99)
            sma_min = min(sma7, sma25, sma99)
            sma_spread = (sma_max - sma_min) / sma_min * 100

            compression_threshold_map = {
                "5m": 0.5, "15m": 0.75, "1h": 1.0, "4h": 1.0, "1d": 1.2
            }
            breakout_range_map = {
                "5m": 0.001, "15m": 0.002, "1h": 0.003, "4h": 0.004, "1d": 0.01
            }
            compression_threshold = compression_threshold_map.get(interval, 1.0)
            breakout_range = breakout_range_map.get(interval, 0.004)

            clustered = sma_spread <= compression_threshold
            price_above_cluster = price > sma_max and (price - sma_max) <= breakout_range * price
            price_below_cluster = price < sma_min and (sma_min - price) <= breakout_range * price

            if trend == "bullish":
                if not (clustered and price_above_cluster):
                    return None
                cluster_label = f"✅ Bullish Breakout ({interval})"
            elif trend == "bearish":
                if not (clustered and price_below_cluster):
                    return None
                cluster_label = f"✅ Bearish Breakout ({interval})"
            else:
                cluster_label = "🔴 No Breakout"

        # === MACD-6 Cross Tracker ===
        macd_series_6 = pd.Series(closes).ewm(span=6).mean() - pd.Series(closes).ewm(span=13).mean()
        signal_series_6 = macd_series_6.ewm(span=6).mean()
        macd_cross_label = "⚪"
        for i in range(1, 30):
            macd_now = macd_series_6.iloc[-i]
            signal_now = signal_series_6.iloc[-i]
            macd_prev = macd_series_6.iloc[-(i + 1)]
            signal_prev = signal_series_6.iloc[-(i + 1)]
            if macd_now > signal_now and macd_prev <= signal_prev:
                macd_cross_label = f"🟢 ({i}c ago)"
                break
            elif macd_now < signal_now and macd_prev >= signal_prev:
                macd_cross_label = f"🔴 ({i}c ago)"
                break

        # === Volume Analysis ===
        volume_trend, volume_price_div = analyze_volume(volumes, closes)

        # === Multi-Timeframe RSI and MACD ===
        rsi_info, macd_info = analyze_multi_tf(symbol, market)

        # === ADX Calculation ===
        adx = calculate_adx(highs, lows, closes)

        # === ATR Calculation ===
        atr_period_map = {"5m": 7, "15m": 7, "1h": 10, "4h": 10, "1d": 10}
        atr_period = atr_period_map.get(interval, 14)
        atr = calculate_atr(highs, lows, closes, period=atr_period)
        if atr < 0.002:
            return None

        atr_dot = "🟢"

        # === Signal Score ===
        score = sum([
            macd_val > signal_val,
            rsi > 52,
            sma_fast > sma_slow,
            volume_trend == "Increasing",
            adx > 25
        ])

        # === Return Result ===
        return (
            symbol, price, sma_pct, rsi, rsi_6, sma_fast, sma_slow,
            sma_third, sma_extra_val, None,
            macd_state, macd_cross_label, volume_trend, volume_price_div,
            rsi_info, macd_info, adx, score, atr, atr_dot, cluster_label
        )

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None



def analyze_volume(volumes, closes):
    avg_vol_7 = pd.Series(volumes).rolling(window=7).mean().iloc[-2]
    avg_vol_5 = pd.Series(volumes).rolling(window=5).mean().iloc[-2]
    volume_trend = "Increasing" if avg_vol_5 > avg_vol_7 else "Decreasing"
    price_trend = "Up" if closes[-1] > closes[-2] else "Down"
    if price_trend == "Up" and volume_trend == "Increasing":
        return volume_trend, "Strong Rally"
    elif price_trend == "Up":
        return volume_trend, "Weak Rally"
    elif price_trend == "Down" and volume_trend == "Increasing":
        return volume_trend, "Strong Sell-off"
    else:
        return volume_trend, "Weak Sell-off"

def get_filtered_results(interval, market, filter_type, trend, sma_min=None, sma_max=None, sma_choice=None):
    symbols = get_usdt_symbols(market)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_symbol, s, interval, market, filter_type, trend, sma_min, sma_max, sma_choice): s
            for s in symbols
        }
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return sorted(results, key=lambda x: abs(x[2]))

# --- Format Results --- 
def format_results(data, interval, market, filter_type=None, trend=None):
    btc_header = get_btc_header()
    filter_info = ""
    if filter_type and trend:
        filter_info = f" - {filter_type.upper()} {trend.title()}"
        header = f"*{interval.upper()} {market.title()} Filtered Results{filter_info}*\n"

    chunks = []
    body = ""
    # Set timezone offset for your region (e.g., +2 for CET)
    TZ_OFFSET = 2
    now_utc = datetime.utcnow()
    local_time = now_utc + timedelta(hours=TZ_OFFSET)
    formatted_time = local_time.strftime('%H:%M')

    for idx, (sym, pr, sma_pct, rsi, rsi_6, sma7, sma25, sma50, sma99, sma200,macd_state, macd_cross_label,
              volume_trend, volume_price_div, rsi_info, macd_info, adx, score, atr, atr_dot, cluster_label) in enumerate(data):
        
        # We only add Multi-TF RSI and MACD for timeframes other than 15m
        rsi_line = ", ".join([f"RSI={r[1]} {r[2]} {r[0]}" if r[1] else f"RSI=NA {r[2]} {r[0]}" for r in rsi_info if r[0] != '15m'])
        macd_line = ", ".join([f"MACD={m[1]} {m[0]}" for m in macd_info if m[0] != '15m'])

        body += f"""
📈 *{sym}({interval}-{formatted_time})*
• SMA Cluster: {cluster_label}
• Price: `${pr:,.4f}`
• SMA% diff: `{sma_pct:+.2f}%`
#RSi-6/MACD-6 used only for scalping--
• RSI-6: `{rsi_6:.1f}` {"🟢" if rsi_6 <= 30 else "🔴" if rsi_6 >= 70 else "⚪"}
• MACD-6: {macd_cross_label}
• Volume Trend: {volume_trend}
• Price/Volume: {volume_price_div}
• ADX Strength: `{adx:.1f}` {"🟢" if adx > 25 else "🔴"}
#Multi TF calculate on the basis of rsi-14 and MACD normal--
• 📊 Trend Score: `{score}/5` {"🟢 Strong" if score >= 4 else "🟡 Moderate" if score == 3 else "🔴 Weak"}
• Multi-TF: {rsi_line}
• Multi-TF: {macd_line}
• • ATR: `{atr:.4f}` {atr_dot}
• SL (LONG): `{pr - (atr * 1.5):.4f}`
• SL (SHORT): `{pr + (atr * 1.5):.4f}`

---------------
"""

    current_chunk = ""
    for section in body.split('\n\n'):
        next_chunk = current_chunk + section + "\n\n"
        if len(btc_header + header + next_chunk) > 4096:
            chunks.append(btc_header + header + current_chunk)
            current_chunk = section + "\n\n"
        else:
            current_chunk = next_chunk
    if current_chunk:
        chunks.append(btc_header + header + current_chunk)

    return chunks


# --- Telegram Bot Handlers ---
def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("SMA% Difference", callback_data='filtertype_smapercent')],
        [InlineKeyboardButton("SMA7/SMA25 Crossover", callback_data='filtertype_smacrossover')],
        [InlineKeyboardButton("SMA7/SMA25/SMA99 Crossover", callback_data='filtertype_sma7x25x99')],
        [InlineKeyboardButton("📐 SMA Cluster Breakout", callback_data='filtertype_smacluster')],
        [InlineKeyboardButton("🗑️ Clear Chat", callback_data='clearchat')],
    ]
    update.message.reply_text(
        "📊 Choose filter type or clear chat:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


def clear_chat(update: Update, context: CallbackContext):
    query = update.callback_query
    chat_id = query.message.chat_id
    bot = context.bot
    for message_id in range(query.message.message_id, query.message.message_id - 100, -1):
        try:
            bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            pass

def set_filter_type(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    filter_type = query.data.split("_")[1]
    user_filters[query.from_user.id] = {'filter_type': filter_type}
    
    if filter_type == "smapercent":
        query.edit_message_text(
            "🔢 Enter SMA% difference range (e.g. 1.5-3):",
            parse_mode=ParseMode.MARKDOWN)
        return
    elif filter_type == "sma7x25x99":
        query.edit_message_text(
            "📏 Select SMA7/SMA25/SMA99 Crossover Setup:",
            parse_mode=ParseMode.MARKDOWN)
        show_bull_bear_buttons(query, filter_type)
        return
    elif filter_type == "smacluster":
        query.edit_message_text(
            "📈 Select trend direction (Bullish or Bearish):",
            parse_mode=ParseMode.MARKDOWN
        )
    show_bull_bear_buttons(query, filter_type)


def handle_sma_percent_input(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_filter = user_filters.get(user_id, {})
    text = update.message.text.strip().replace("%", "").replace("to", "-").replace("–", "-")
    try:
        if "-" not in text: raise ValueError
        min_str, max_str = text.split("-", 1)
        sma_min = float(min_str.strip())
        sma_max = float(max_str.strip())
        if not (0.01 <= sma_min < sma_max <= 50): raise ValueError
        user_filter["sma_min"] = sma_min
        user_filter["sma_max"] = sma_max
        user_filters[user_id] = user_filter
        keyboard = [
            [InlineKeyboardButton("SMA5", callback_data='smaoption_sma5')],
            [InlineKeyboardButton("SMA7", callback_data='smaoption_sma7')],
            [InlineKeyboardButton("SMA25", callback_data='smaoption_sma25')],
            [InlineKeyboardButton("SMA50", callback_data='smaoption_sma50')],
            [InlineKeyboardButton("SMA99", callback_data='smaoption_sma99')],
            [InlineKeyboardButton("SMA200", callback_data='smaoption_sma200')],
        ]
        update.message.reply_text("📏 Select SMA for % difference:", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception:
        update.message.reply_text("❌ Invalid range. Use format like `1.5-3`.")

def handle_sma_option(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    sma_choice = query.data.split("_")[1]
    user_filter = user_filters.get(query.from_user.id, {})
    user_filter["sma_choice"] = sma_choice
    user_filters[query.from_user.id] = user_filter
    show_bull_bear_buttons(query, user_filter["filter_type"])

def show_bull_bear_buttons(ref, filter_type, is_message=False):
    keyboard = [
        [InlineKeyboardButton("Bullish Setup", callback_data='setup_bullish')],
        [InlineKeyboardButton("Bearish Setup", callback_data='setup_bearish')]
    ]
    text = f"✅ Filter type set to *{filter_type.title()}*. Choose setup:"
    if is_message:
        ref.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        ref.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

def set_trend(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    trend = query.data.split("_")[1]
    user_filter = user_filters.get(query.from_user.id, {})
    user_filter['trend'] = trend
    user_filters[query.from_user.id] = user_filter
    keyboard = [
    [InlineKeyboardButton("5m Spot", callback_data='5m_spot'), InlineKeyboardButton("15m Spot", callback_data='15m_spot'),
     InlineKeyboardButton("1h Spot", callback_data='1h_spot'), InlineKeyboardButton("4h Spot", callback_data='4h_spot'), InlineKeyboardButton("1d Spot", callback_data='1d_spot')],
    [InlineKeyboardButton("5m Futures", callback_data='5m_futures'), InlineKeyboardButton("15m Futures", callback_data='15m_futures'),
     InlineKeyboardButton("1h Futures", callback_data='1h_futures'), InlineKeyboardButton("4h Futures", callback_data='4h_futures'), InlineKeyboardButton("1d Futures", callback_data='1d_futures')]
]


    query.edit_message_text(f"✅ Setup set to *{trend.title()}*. Choose timeframe & market:", parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

def handle_analysis(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    interval, market = query.data.split("_")
    user_filter = user_filters.get(query.from_user.id, {})
    filter_type = user_filter.get('filter_type')
    trend = user_filter.get('trend')
    sma_min = user_filter.get('sma_min')
    sma_max = user_filter.get('sma_max')
    sma_choice = user_filter.get('sma_choice')

    if not filter_type or not trend:
        query.edit_message_text("❌ Start with /start to select filter and setup.")
        return

    query.edit_message_text(f"⏳ Fetching {trend.title()} {filter_type} results for {interval.upper()} {market.title()}...")
    results = get_filtered_results(interval, market, filter_type, trend, sma_min, sma_max, sma_choice)

    if not results:
        query.message.reply_text(f"❌ No matches for {interval.upper()} {market} [{trend}].")
        return

    messages = format_results(results, interval, market, filter_type, trend)
    for msg in messages:
        query.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

def refresh(update: Update, context: CallbackContext):
    get_usdt_symbols.cache_clear()
    update.message.reply_text("🔄 Cache cleared!")

def main():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("refresh", refresh))
    dp.add_handler(CallbackQueryHandler(clear_chat, pattern="^clearchat$"))
    dp.add_handler(CallbackQueryHandler(set_filter_type, pattern="^filtertype_"))
    dp.add_handler(CallbackQueryHandler(handle_sma_option, pattern="^smaoption_"))
    dp.add_handler(CallbackQueryHandler(set_trend, pattern="^setup_"))
    dp.add_handler(CallbackQueryHandler(handle_analysis, pattern="^(5m|15m|1h|4h|1d)_(spot|futures)$"))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_sma_percent_input))

    print("🤖 Bot is running...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
