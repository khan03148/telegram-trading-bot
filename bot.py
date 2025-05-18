# --- Existing imports ---
import requests
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler, MessageHandler, Filters
from functools import lru_cache
import concurrent.futures

# --- Constants ---
TELEGRAM_BOT_TOKEN = "7575210959:AAFzqsnt7Gx3qMBxl4M9gzOFNsAFSKn2W2Q"
TELEGRAM_CHAT_ID = "906015851"
MAX_WORKERS = 12
session = requests.Session()
user_filters = {}

# --- Utility Functions ---
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
    return session.get(url, params={"symbol": symbol, "interval": interval, "limit": 100}).json()

def calculate_macd(close_prices):
    short_ema = pd.Series(close_prices).ewm(span=12).mean()
    long_ema = pd.Series(close_prices).ewm(span=26).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9).mean()
    return macd.iloc[-1], signal.iloc[-1]

def compute_rsi(closes, period=14):
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

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

def analyze_multi_tf(symbol, market):
    tf_map = [("1h", "1h"), ("4h", "4h"), ("1d", "1d")]
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

def analyze_volume(volumes, closes):
    avg_vol_10 = pd.Series(volumes).rolling(window=10).mean().iloc[-2]
    avg_vol_5 = pd.Series(volumes).rolling(window=5).mean().iloc[-2]
    volume_trend = "Increasing" if avg_vol_5 > avg_vol_10 else "Decreasing"
    price_trend = "Up" if closes[-1] > closes[-2] else "Down"
    if price_trend == "Up" and volume_trend == "Increasing":
        return volume_trend, "Strong Rally"
    elif price_trend == "Up":
        return volume_trend, "Weak Rally"
    elif price_trend == "Down" and volume_trend == "Increasing":
        return volume_trend, "Strong Sell-off"
    else:
        return volume_trend, "Weak Sell-off"

def process_symbol(symbol, interval, market, filter_type, trend, sma_min=None, sma_max=None, sma_choice=None):
    try:
        klines = fetch_klines(symbol, interval, market)
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        if len(closes) < 50: return None

        sma7 = pd.Series(closes).rolling(window=7).mean().iloc[-1]
        sma25 = pd.Series(closes).rolling(window=25).mean().iloc[-1]
        sma50 = pd.Series(closes).rolling(window=50).mean().iloc[-1]
        sma99 = pd.Series(closes).rolling(window=99).mean().iloc[-1]
        sma200 = pd.Series(closes).rolling(window=200).mean().iloc[-1]
        sma7_prev = pd.Series(closes).rolling(window=7).mean().iloc[-2]
        sma25_prev = pd.Series(closes).rolling(window=25).mean().iloc[-2]

        price = closes[-1]
        sma_value = {'sma7': sma7, 'sma25': sma25, 'sma50': sma50, 'sma99': sma99, 'sma200': sma200}.get(sma_choice, sma25)
        sma_pct = (price - sma_value) / sma_value * 100
        rsi = compute_rsi(closes)

        if filter_type == "smapercent":
            if sma_min is None or sma_max is None: return None
            if trend == "bullish" and not (sma_min <= sma_pct <= sma_max and rsi > 52): return None
            if trend == "bearish" and not (-sma_max <= sma_pct <= -sma_min and rsi < 48): return None
        elif filter_type == "smacrossover":
            if trend == "bullish" and not (sma7 > sma25 and sma7_prev <= sma25_prev and rsi > 52): return None
            if trend == "bearish" and not (sma7 < sma25 and sma7_prev >= sma25_prev and rsi < 48): return None

        macd_val, signal_val = calculate_macd(closes)
        macd_state = "🟢 Bullish" if macd_val > signal_val else "🔴 Bearish"
        volume_trend, volume_price_div = analyze_volume(volumes, closes)
        rsi_info, macd_info = analyze_multi_tf(symbol, market)
        adx = calculate_adx(highs, lows, closes)
        score = sum([
            macd_val > signal_val,
            rsi > 52,
            sma7 > sma25,
            volume_trend == "Increasing",
            adx > 25
        ])
        return (symbol, price, sma_pct, rsi, sma7, sma25, sma50, sma99, sma200,
                macd_state, volume_trend, volume_price_div, rsi_info, macd_info, adx, score)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

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

def format_results(data, interval, market):
    header = f"*{interval.upper()} {market.title()} Filtered Results*\n"
    chunks = []
    body = ""
    for (sym, pr, sma_pct, rsi, sma7, sma25, sma50, sma99, sma200, macd_state,
         volume_trend, volume_price_div, rsi_info, macd_info, adx, score) in data:
        rsi_line = ", ".join([f"RSI={r[1]} {r[2]} {r[0]}" if r[1] else f"RSI=NA {r[2]} {r[0]}" for r in rsi_info])
        macd_line = ", ".join([f"MACD={m[1]} {m[0]}" for m in macd_info])
        body += f"""
📈 *{sym}*
• Price: `${pr:,.4f}`
• SMA% from SMA25: `{sma_pct:+.2f}%`
• RSI14: `{rsi:.1f}` {"🟢" if rsi >= 52 else "🔴" if rsi <= 48 else "⚪"}
• MACD: {macd_state}
• Volume Trend: {volume_trend}
• Price/Volume: {volume_price_div}
• ADX Strength: `{adx:.1f}` {"🟢" if adx > 25 else "🔴"}
• 📊 Trend Score: `{score}/5` {"🟢 Strong" if score >= 4 else "🟡 Moderate" if score == 3 else "🔴 Weak"}
• Multi-TF RSI: {rsi_line}
• Multi-TF MACD: {macd_line}
"""

    for line in body.split('\n\n'):
        if len(header + line) > 4096:
            chunks.append(header)
            header = line
        else:
            header += line + "\n\n"
    if header:
        chunks.append(header)
    return chunks
# --- Telegram Bot Interaction ---

def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("SMA% Difference", callback_data='filtertype_smapercent')],
        [InlineKeyboardButton("SMA7/SMA25 Crossover", callback_data='filtertype_smacrossover')],
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
            "🔢 Please enter your desired SMA% difference range (e.g. 1.5-3):",
            parse_mode=ParseMode.MARKDOWN)
        return
    show_bull_bear_buttons(query, filter_type)

def handle_sma_percent_input(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_filter = user_filters.get(user_id, {})
    text = update.message.text.strip().replace("%", "").replace("to", "-").replace("–", "-")
    try:
        if "-" not in text:
            raise ValueError
        min_str, max_str = text.split("-", 1)
        sma_min = float(min_str.strip())
        sma_max = float(max_str.strip())
        if not (0.01 <= sma_min < sma_max <= 50):
            raise ValueError
        user_filter["sma_min"] = sma_min
        user_filter["sma_max"] = sma_max
        user_filters[user_id] = user_filter
        keyboard = [
            [InlineKeyboardButton("SMA7", callback_data='smaoption_sma7')],
            [InlineKeyboardButton("SMA25", callback_data='smaoption_sma25')],
            [InlineKeyboardButton("SMA50", callback_data='smaoption_sma50')],
            [InlineKeyboardButton("SMA99", callback_data='smaoption_sma99')],
            [InlineKeyboardButton("SMA200", callback_data='smaoption_sma200')],
        ]
        update.message.reply_text(
            "📏 Select which SMA to use for the % difference:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    except Exception:
        update.message.reply_text("❌ Please enter a valid SMA% range like `1.5-3` (min-max), where min < max.")

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
    text = f"✅ Filter type set to *{filter_type.title()}*.\nChoose setup type:"
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
        [InlineKeyboardButton("1h Spot", callback_data='1h_spot'), InlineKeyboardButton("4h Spot", callback_data='4h_spot'), InlineKeyboardButton("1d Spot", callback_data='1d_spot')],
        [InlineKeyboardButton("1h Futures", callback_data='1h_futures'), InlineKeyboardButton("4h Futures", callback_data='4h_futures'), InlineKeyboardButton("1d Futures", callback_data='1d_futures')]
    ]
    query.edit_message_text(f"✅ Setup set to *{trend.title()}*.\nChoose timeframe and market:", parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

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
        query.edit_message_text("❌ Please start with /start to choose filter and setup.")
        return

    query.edit_message_text(f"⏳ Fetching {trend.title()} {filter_type} results for {interval.upper()} {market.title()}...")
    results = get_filtered_results(interval, market, filter_type, trend, sma_min, sma_max, sma_choice)

    if not results:
        query.message.reply_text(f"❌ No matching results for {interval.upper()} ({market}) [{trend}].")
        return

    messages = format_results(results, interval, market)
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
    dp.add_handler(CallbackQueryHandler(handle_analysis, pattern="^(1h|4h|1d)_(spot|futures)$"))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_sma_percent_input))

    print("🤖 Bot is running...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
