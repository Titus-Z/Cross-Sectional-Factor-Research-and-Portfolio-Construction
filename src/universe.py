"""股票池定义模块。

这个文件把“默认下载哪些股票”这件事从 `main.py` 里拆出来，
目的是让主流程更清晰，也方便以后继续扩展不同股票池。

当前提供两个硬编码的大盘股票池，并支持从 `data/universe/` 读取动态股票池：

- `us_large_cap_100`
- `us_large_cap_300`
- `us_active_3000`

它由 100 只美股大盘股组成，覆盖科技、金融、医疗、消费、工业、能源等主要板块。

为什么要扩大股票池？

1. 之前只有 12 只股票，横截面太小，IC 和分组收益波动会非常大；
2. 当天可排序的股票太少时，模型很容易“看起来有点信号”，但统计上并不稳；
3. 扩到 100 只以上以后，横截面指标会更接近真实量化研究环境。
4. 继续扩到 300 只以后，更容易判断“之前结果差”到底是不是因为股票池太小、横截面太窄。

另外，这里还顺手维护了一份“股票 -> 行业/板块”映射，
后续横截面中性化步骤会直接复用它。
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

US_LARGE_CAP_100 = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO",
    "QCOM", "IBM", "NOW", "AMAT", "MU", "TXN", "INTU", "ADP", "ANET", "PANW",
    "SNPS", "KLAC", "CDNS", "ACN",
    "META", "GOOGL", "GOOG", "NFLX", "TMUS", "VZ", "T", "DIS",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "ROST",
    "CMG", "MAR",
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "MDLZ",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "PGR",
    "MMC", "USB", "BK", "MA",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "DHR", "ABT", "BMY",
    "MDT", "ISRG", "AMGN",
    "GE", "RTX", "HON", "CAT", "DE", "UNP", "UPS", "BA", "LMT", "ETN", "WM",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "LIN", "APD",
    "NEE",
    "AMT",
]


# 这个 300 股票池来自当前 S&P 500 中按指数权重排序的前 300 只成分股，
# 再统一转换成 yfinance 可识别的 ticker 形式（例如 `BRK.B -> BRK-B`）。
#
# 这样做的理由很务实：
#
# 1. 指数权重本身就是“大盘股”的直接代理；
# 2. 能进入这 300 只的股票通常也具备较好的成交活跃度；
# 3. 这个口径足够稳定，能让我们更专注于研究“股票池变大以后，结果会不会更稳”。
US_LARGE_CAP_300 = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "META", "TSLA", "BRK-B",
    "WMT", "LLY", "JPM", "XOM", "JNJ", "V", "MA", "COST", "ORCL", "CVX",
    "NFLX", "MU", "ABBV", "PLTR", "BAC", "PG", "AMD", "CAT", "KO", "HD",
    "CSCO", "GE", "MRK", "AMAT", "LRCX", "MS", "RTX", "PM", "GS", "WFC",
    "UNH", "GEV", "TMUS", "LIN", "IBM", "INTC", "MCD", "PEP", "VZ", "AXP",
    "T", "C", "KLAC", "NEE", "AMGN", "TMO", "ABT", "TJX", "TXN", "GILD",
    "CRM", "DIS", "ISRG", "SCHW", "COP", "PFE", "BA", "APH", "ADI", "ANET",
    "DE", "BLK", "UBER", "UNP", "HON", "LMT", "ETN", "WELL", "QCOM", "APP",
    "DHR", "BKNG", "LOW", "PANW", "SPGI", "SYK", "CB", "BMY", "PLD", "ACN",
    "INTU", "GLW", "NEM", "PGR", "VRTX", "COF", "PH", "MDT", "MO", "NOW",
    "SO", "HCA", "CME", "DELL", "MCK", "CMCSA", "DUK", "SBUX", "CEG", "CRWD",
    "ADBE", "NOC", "VRT", "EQIX", "SNDK", "BSX", "WM", "GD", "HWM", "WDC",
    "TT", "CVS", "BX", "ICE", "WMB", "STX", "MAR", "FCX", "FDX", "MRSH",
    "UPS", "PNC", "PWR", "KKR", "ADP", "BK", "REGN", "JCI", "USB", "AMT",
    "SHW", "SLB", "ORLY", "MCO", "CDNS", "EOG", "CSX", "SNPS", "MMM", "ABNB",
    "ECL", "ITW", "CMI", "KMI", "RCL", "EMR", "VLO", "MDLZ", "PSX", "MSI",
    "MNST", "MPC", "AEP", "NKE", "CI", "HLT", "ROST", "CRH", "AON", "WBD",
    "CL", "RSG", "CTAS", "GM", "TDG", "DASH", "LHX", "APD", "ELV", "NSC",
    "APO", "SRE", "OXY", "HOOD", "TRV", "DLR", "PCAR", "SPG", "COR", "BKR",
    "TEL", "FTNT", "O", "TFC", "AFL", "AJG", "CTVA", "OKE", "AZO", "CIEN",
    "FANG", "TGT", "D", "MPWR", "ALL", "TRGP", "FAST", "GWW", "EA", "LITE",
    "VST", "ETR", "ADSK", "KEYS", "EXC", "NXPI", "ZTS", "XEL", "CAH", "AME",
    "FIX", "NDAQ", "PSA", "CARR", "TER", "F", "COIN", "EW", "MET", "URI",
    "COHR", "CVNA", "IDXX", "BDX", "GRMN", "KR", "DAL", "YUM", "WAB", "HSY",
    "FITB", "DDOG", "CMG", "PYPL", "ED", "CBRE", "EBAY", "ODFL", "PEG", "AIG",
    "ROK", "AMP", "MSCI", "DHI", "EQT", "VTR", "NUE", "PCG", "WEC", "HIG",
    "TTWO", "ROP", "LVS", "XYZ", "CCL", "LYV", "KDP", "CCI", "VMC", "STT",
    "ADM", "MCHP", "ACGL", "AXON", "SYY", "SATS", "MLM", "PRU", "WDAY", "KVUE",
    "PAYX", "EME", "TPL", "RMD", "GEHC", "HAL", "A", "CPRT", "KMB", "HBAN",
    "HPE", "IR", "NRG", "DVN", "MTB", "ATO", "IRM", "AEE", "DTE", "OTIS",
]


SYMBOL_TO_SECTOR = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "AMD": "Information Technology",
    "INTC": "Information Technology",
    "CSCO": "Information Technology",
    "QCOM": "Information Technology",
    "IBM": "Information Technology",
    "NOW": "Information Technology",
    "AMAT": "Information Technology",
    "MU": "Information Technology",
    "TXN": "Information Technology",
    "INTU": "Information Technology",
    "ADP": "Information Technology",
    "ANET": "Information Technology",
    "PANW": "Information Technology",
    "SNPS": "Information Technology",
    "KLAC": "Information Technology",
    "CDNS": "Information Technology",
    "ACN": "Information Technology",
    "META": "Communication Services",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "NFLX": "Communication Services",
    "TMUS": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "DIS": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "ROST": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PM": "Consumer Staples",
    "MO": "Consumer Staples",
    "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "AXP": "Financials",
    "PGR": "Financials",
    "MMC": "Financials",
    "USB": "Financials",
    "BK": "Financials",
    "MA": "Financials",
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "PFE": "Health Care",
    "TMO": "Health Care",
    "DHR": "Health Care",
    "ABT": "Health Care",
    "BMY": "Health Care",
    "MDT": "Health Care",
    "ISRG": "Health Care",
    "AMGN": "Health Care",
    "GE": "Industrials",
    "RTX": "Industrials",
    "HON": "Industrials",
    "CAT": "Industrials",
    "DE": "Industrials",
    "UNP": "Industrials",
    "UPS": "Industrials",
    "BA": "Industrials",
    "LMT": "Industrials",
    "ETN": "Industrials",
    "WM": "Industrials",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "LIN": "Materials",
    "APD": "Materials",
    "NEE": "Utilities",
    "AMT": "Real Estate",
    "BRK-B": "Financials",
    "PLTR": "Information Technology",
    "GEV": "Industrials",
    "APP": "Information Technology",
    "WELL": "Real Estate",
    "SPGI": "Financials",
    "GLW": "Information Technology",
    "UBER": "Industrials",
    "SNDK": "Information Technology",
    "VRT": "Industrials",
    "HWM": "Industrials",
    "WDC": "Information Technology",
    "TT": "Industrials",
    "BX": "Financials",
    "ICE": "Financials",
    "WMB": "Energy",
    "STX": "Information Technology",
    "FCX": "Materials",
    "MRSH": "Financials",
    "PWR": "Industrials",
    "KKR": "Financials",
    "REGN": "Health Care",
    "SHW": "Materials",
    "ORLY": "Consumer Discretionary",
    "MCO": "Financials",
    "CSX": "Industrials",
    "MMM": "Industrials",
    "ABNB": "Consumer Discretionary",
    "ECL": "Materials",
    "ITW": "Industrials",
    "CMI": "Industrials",
    "KMI": "Energy",
    "RCL": "Consumer Discretionary",
    "EMR": "Industrials",
    "VLO": "Energy",
    "PSX": "Energy",
    "MSI": "Information Technology",
    "MNST": "Consumer Staples",
    "MPC": "Energy",
    "AEP": "Utilities",
    "CI": "Health Care",
    "HLT": "Consumer Discretionary",
    "CRH": "Materials",
    "AON": "Financials",
    "WBD": "Communication Services",
    "RSG": "Industrials",
    "CTAS": "Industrials",
    "GM": "Consumer Discretionary",
    "TDG": "Industrials",
    "DASH": "Consumer Discretionary",
    "LHX": "Industrials",
    "ELV": "Health Care",
    "NSC": "Industrials",
    "APO": "Financials",
    "SRE": "Utilities",
    "OXY": "Energy",
    "HOOD": "Financials",
    "TRV": "Financials",
    "DLR": "Real Estate",
    "PCAR": "Industrials",
    "SPG": "Real Estate",
    "COR": "Health Care",
    "BKR": "Energy",
    "TEL": "Information Technology",
    "FTNT": "Information Technology",
    "O": "Real Estate",
    "TFC": "Financials",
    "AFL": "Financials",
    "AJG": "Financials",
    "CTVA": "Materials",
    "OKE": "Energy",
    "AZO": "Consumer Discretionary",
    "CIEN": "Information Technology",
    "FANG": "Energy",
    "D": "Utilities",
    "MPWR": "Information Technology",
    "ALL": "Financials",
    "TRGP": "Energy",
    "FAST": "Industrials",
    "GWW": "Industrials",
    "EA": "Communication Services",
    "LITE": "Information Technology",
    "VST": "Utilities",
    "ETR": "Utilities",
    "ADSK": "Information Technology",
    "KEYS": "Information Technology",
    "EXC": "Utilities",
    "NXPI": "Information Technology",
    "ZTS": "Health Care",
    "XEL": "Utilities",
    "CAH": "Health Care",
    "AME": "Industrials",
    "FIX": "Industrials",
    "NDAQ": "Financials",
    "PSA": "Real Estate",
    "CARR": "Industrials",
    "TER": "Information Technology",
    "F": "Consumer Discretionary",
    "COIN": "Financials",
    "EW": "Health Care",
    "MET": "Financials",
    "URI": "Industrials",
    "COHR": "Information Technology",
    "CVNA": "Consumer Discretionary",
    "IDXX": "Health Care",
    "BDX": "Health Care",
    "GRMN": "Consumer Discretionary",
    "KR": "Consumer Staples",
    "DAL": "Industrials",
    "YUM": "Consumer Discretionary",
    "WAB": "Industrials",
    "HSY": "Consumer Staples",
    "FITB": "Financials",
    "DDOG": "Information Technology",
    "PYPL": "Financials",
    "ED": "Utilities",
    "CBRE": "Real Estate",
    "EBAY": "Consumer Discretionary",
    "ODFL": "Industrials",
    "PEG": "Utilities",
    "AIG": "Financials",
    "ROK": "Industrials",
    "AMP": "Financials",
    "MSCI": "Financials",
    "DHI": "Consumer Discretionary",
    "EQT": "Energy",
    "VTR": "Real Estate",
    "NUE": "Materials",
    "PCG": "Utilities",
    "WEC": "Utilities",
    "HIG": "Financials",
    "TTWO": "Communication Services",
    "ROP": "Information Technology",
    "LVS": "Consumer Discretionary",
    "XYZ": "Financials",
    "CCL": "Consumer Discretionary",
    "LYV": "Communication Services",
    "KDP": "Consumer Staples",
    "CCI": "Real Estate",
    "VMC": "Materials",
    "STT": "Financials",
    "ADM": "Consumer Staples",
    "MCHP": "Information Technology",
    "ACGL": "Financials",
    "AXON": "Industrials",
    "SYY": "Consumer Staples",
    "SATS": "Communication Services",
    "MLM": "Materials",
    "PRU": "Financials",
    "WDAY": "Information Technology",
    "KVUE": "Consumer Staples",
    "PAYX": "Industrials",
    "EME": "Industrials",
    "TPL": "Energy",
    "RMD": "Health Care",
    "GEHC": "Health Care",
    "HAL": "Energy",
    "A": "Health Care",
    "CPRT": "Industrials",
    "KMB": "Consumer Staples",
    "HBAN": "Financials",
    "HPE": "Information Technology",
    "IR": "Industrials",
    "NRG": "Utilities",
    "DVN": "Energy",
    "MTB": "Financials",
    "ATO": "Utilities",
    "IRM": "Real Estate",
    "AEE": "Utilities",
    "DTE": "Utilities",
    "OTIS": "Industrials",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_UNIVERSE_FILES = {
    "us_active_3000": PROJECT_ROOT / "data" / "universe" / "us_active_3000_symbols.csv",
}

UNIVERSE_DEFINITIONS = {
    "us_large_cap_100": US_LARGE_CAP_100,
    "us_large_cap_300": US_LARGE_CAP_300,
}


@lru_cache(maxsize=8)
def _load_universe_symbols_from_csv(path: str) -> tuple[str, ...]:
    """从 data/universe 的 CSV 文件读取动态股票池。"""

    csv_path = Path(path)
    if not csv_path.exists():
        return tuple()
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        symbols = [
            str(row.get("instrument_id", "")).strip()
            for row in reader
            if str(row.get("instrument_id", "")).strip()
        ]
    return tuple(symbols)


def list_supported_universes() -> list[str]:
    """列出当前代码内置支持的股票池名字。"""

    names = list(UNIVERSE_DEFINITIONS.keys())
    for name, path in DYNAMIC_UNIVERSE_FILES.items():
        if path.exists():
            names.append(name)
    return names


def get_universe_symbols(universe_name: str) -> list[str]:
    """根据股票池名字返回股票代码列表。"""

    normalized_name = universe_name.strip().lower()
    if normalized_name not in UNIVERSE_DEFINITIONS:
        if normalized_name in DYNAMIC_UNIVERSE_FILES:
            symbols = _load_universe_symbols_from_csv(str(DYNAMIC_UNIVERSE_FILES[normalized_name]))
            if symbols:
                return list(symbols)
        raise ValueError(
            f"Unsupported universe name: {universe_name}. "
            f"Supported universes: {list_supported_universes()}"
        )
    return list(UNIVERSE_DEFINITIONS[normalized_name])


def get_symbol_sector_map(symbols: list[str]) -> dict[str, str]:
    """返回股票代码到板块的映射。

    这里用的是较粗粒度的 GICS 风格板块信息。
    在横截面中性化时，这已经足够比“完全没有行业信息”更实用。
    """

    return {symbol: SYMBOL_TO_SECTOR[symbol] for symbol in symbols if symbol in SYMBOL_TO_SECTOR}
