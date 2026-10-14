import logging
import math
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="실시간 미세먼지 검색", layout="wide")


# --- API 키: 코드에 직접 넣지 않고 secrets / 환경변수에서 읽기 ---
def load_key(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


KAKAO_KEY = load_key("KAKAO_KEY")
AIR_KEY = load_key("AIR_KEY")

if not KAKAO_KEY or not AIR_KEY:
    st.error(
        "API 키가 설정되지 않았습니다. `.streamlit/secrets.toml`에 "
        "`KAKAO_KEY`, `AIR_KEY`를 등록하거나 환경변수로 설정해주세요."
    )
    st.stop()

# --- 상수 ---
AIR_BASE = "https://apis.data.go.kr/B552584"
KAKAO_BASE = "https://dapi.kakao.com/v2/local"
STALE_MINUTES = 180  # 측정값이 이 시간(분)보다 오래되면 지연 경고
MAX_APPROX_KM = 25   # 공식 근접 측정소가 없을 때 대체할 최대 거리(km)
MAP_HEIGHT = 420     # 지도 높이(px)

# 카카오(전체 이름) -> 에어코리아(약칭) 시도명 매핑
SIDO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
    "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
    "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
    "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북",
    "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
    "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
}
SIDO_SET = set(SIDO_SHORT.values())

# (등급, 이모지, 배경색, 글자색, RGB)
GRADE_META = [
    ("좋음", "🔵", "#1E88E5", "#FFFFFF", (30, 136, 229)),
    ("보통", "🟢", "#43A047", "#FFFFFF", (67, 160, 71)),
    ("나쁨", "🟡", "#FDD835", "#3E2F00", (253, 216, 53)),
    ("매우 나쁨", "🔴", "#E53935", "#FFFFFF", (229, 57, 53)),
]
UNKNOWN_META = ("점검 중", "⚪", "#808080", "#FFFFFF", (128, 128, 128))

# 등급 상한값 (좋음 / 보통 / 나쁨 상한, 그 이상은 매우 나쁨)
THRESHOLDS = {"PM10": (30, 80, 150), "PM2.5": (15, 35, 75)}


# --- 네트워크 세션 (연결 재사용으로 속도 향상) ---
@st.cache_resource
def get_session():
    return requests.Session()


# --- 유틸 ---
def normalize_sido(name):
    """'충청북도', '충북', '경기 수원시' 등 어떤 형태든 에어코리아 약칭으로 변환"""
    if not name:
        return None
    first = name.split()[0]
    if first in SIDO_SHORT:
        return SIDO_SHORT[first]
    if first in SIDO_SET:
        return first
    return first[:2]


def get_grade(value, kind="PM10"):
    if value is None or pd.isna(value):
        return UNKNOWN_META
    for limit, meta in zip(THRESHOLDS[kind], GRADE_META):
        if value <= limit:
            return meta
    return GRADE_META[-1]


def badge(meta, label=None, size="0.9rem"):
    """등급을 색 배지로 표시"""
    text, emoji, bg, fg, _ = meta
    shown = label or f"{emoji} {text}"
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 10px;"
        f"border-radius:999px;font-weight:600;font-size:{size};"
        f"white-space:nowrap'>{shown}</span>"
    )


def render_legend():
    def row(kind, name):
        limits = THRESHOLDS[kind]
        ranges = [
            f"0~{limits[0]}",
            f"{limits[0] + 1}~{limits[1]}",
            f"{limits[1] + 1}~{limits[2]}",
            f"{limits[2] + 1}~",
        ]
        chips = " ".join(
            badge(meta, f"{meta[0]} {r}", size="0.8rem")
            for meta, r in zip(GRADE_META, ranges)
        )
        return f"<div style='margin:6px 0'><b>{name}</b> &nbsp; {chips}</div>"

    with st.expander("📖 등급 기준 보기 (단위: ㎍/㎥)"):
        st.markdown(
            row("PM10", "미세먼지 (PM10)") + row("PM2.5", "초미세먼지 (PM2.5)"),
            unsafe_allow_html=True,
        )


def now_kst():
    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)


def minutes_since(ts):
    return int((now_kst() - ts).total_seconds() // 60)


def ago_text(ts):
    if ts is None or pd.isna(ts):
        return "알 수 없음"
    minutes = minutes_since(ts)
    if minutes < 1:
        return "방금 전"
    if minutes < 60:
        return f"{minutes}분 전"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}시간 {mins}분 전" if mins else f"{hours}시간 전"
    return f"{hours // 24}일 전"


def api_get(url, retries=1, timeout=(3, 5), **kwargs):
    """타임아웃 단축 + 세션 재사용으로 빠른 JSON 요청"""
    session = get_session()
    last_error = None
    for attempt in range(retries + 1):
        try:
            res = session.get(url, timeout=timeout, **kwargs)
            res.raise_for_status()
            return res.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.3)
    raise last_error


def describe_error(e):
    """화면에 보여줄 짧은 오류 사유"""
    if isinstance(e, requests.Timeout):
        return "응답 지연"
    if isinstance(e, requests.ConnectionError):
        return "연결 실패"
    if isinstance(e, requests.HTTPError):
        return "서버 오류"
    return type(e).__name__


def parse_data_time(series):
    """에어코리아의 '24:00'은 '다음날 00:00'이므로 하루를 더해서 변환"""
    s = series.astype(str)
    is24 = s.str.endswith("24:00")
    dt = pd.to_datetime(s.str.replace("24:00", "00:00", regex=False), errors="coerce")
    dt.loc[is24] = dt.loc[is24] + pd.Timedelta(days=1)
    return dt


# --- 데이터 호출 함수 (show_spinner=False로 초록색 함수명 알림창 제거) ---
@st.cache_data(ttl=600, show_spinner=False)
def get_sido_air_quality():
    data = api_get(
        f"{AIR_BASE}/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty",
        params={
            "serviceKey": AIR_KEY,
            "returnType": "json",
            "numOfRows": 1000,
            "pageNo": 1,
            "sidoName": "전국",
            "ver": "1.0",
        },
        timeout=(3, 8),
    )
    items = data.get("response", {}).get("body", {}).get("items", [])
    df = pd.DataFrame(items)
    empty = {"df": pd.DataFrame(), "data_time": None, "fetched_at": now_kst()}
    if df.empty or "pm10Value" not in df.columns:
        return empty

    df["pm10Value"] = pd.to_numeric(df["pm10Value"], errors="coerce")
    df = df.dropna(subset=["pm10Value"])
    if df.empty:
        return empty

    data_time = None
    if "dataTime" in df.columns:
        latest = parse_data_time(df["dataTime"]).max()
        data_time = None if pd.isna(latest) else latest

    sido_avg = df.groupby("sidoName")["pm10Value"].mean().reset_index()
    return {
        "df": sido_avg.sort_values(by="pm10Value"),
        "data_time": data_time,
        "fetched_at": now_kst(),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def get_kakao_candidates(query):
    """검색 후보 목록 (주소 검색 우선, 없으면 장소명 검색)"""
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {"query": query, "size": 10}
    cands = {}

    docs = api_get(
        f"{KAKAO_BASE}/search/address.json", headers=headers, params=params
    ).get("documents") or []
    for doc in docs:
        info = doc.get("address")
        sido = info["region_1depth_name"] if info else doc["address_name"]
        label = f"📍 {doc['address_name']}"
        cands.setdefault(
            label,
            {"label": label, "x": float(doc["x"]), "y": float(doc["y"]),
             "sido": normalize_sido(sido)},
        )

    if not cands:
        docs = api_get(
            f"{KAKAO_BASE}/search/keyword.json", headers=headers, params=params
        ).get("documents") or []
        for doc in docs:
            addr = doc.get("road_address_name") or doc.get("address_name") or ""
            label = f"{doc['place_name']} · {addr}"
            cands.setdefault(
                label,
                {"label": label, "x": float(doc["x"]), "y": float(doc["y"]),
                 "sido": normalize_sido(doc.get("address_name"))},
            )

    return list(cands.values())


@st.cache_data(ttl=86400, show_spinner=False)
def convert_to_tm(x, y):
    docs = api_get(
        f"{KAKAO_BASE}/geo/transcoord.json",
        headers={"Authorization": f"KakaoAK {KAKAO_KEY}"},
        params={"x": x, "y": y, "output_coord": "TM"},
    ).get("documents")
    if docs:
        return docs[0]["x"], docs[0]["y"]
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_nearby_stations(tmX, tmY):
    """공식 근접 측정소 후보 목록"""
    data = api_get(
        f"{AIR_BASE}/MsrstnInfoInqireSvc/getNearbyMsrstnList",
        params={"serviceKey": AIR_KEY, "returnType": "json", "tmX": tmX, "tmY": tmY},
    )
    items = data.get("response", {}).get("body", {}).get("items") or []
    return [
        {"name": i["stationName"], "distance": i.get("tm"), "addr": i.get("addr")}
        for i in items
        if isinstance(i, dict) and i.get("stationName")
    ]


@st.cache_data(ttl=86400, show_spinner=False)
def get_station_location(station_name, addr=None):
    """측정소의 (위도, 경도)"""
    data = api_get(
        f"{AIR_BASE}/MsrstnInfoInqireSvc/getMsrstnList",
        params={
            "serviceKey": AIR_KEY,
            "returnType": "json",
            "numOfRows": 20,
            "pageNo": 1,
            "stationName": station_name,
        },
    )
    items = data.get("response", {}).get("body", {}).get("items") or []
    cands = [i for i in items if i.get("stationName") == station_name] or items
    if addr:
        cands = [i for i in cands if i.get("addr") == addr] or cands
    for item in cands:
        try:
            lat, lon = float(item["dmX"]), float(item["dmY"])
        except (KeyError, TypeError, ValueError):
            continue
        if lat > 60 and lon < 60:
            lat, lon = lon, lat
        return lat, lon
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_all_stations():
    """전국 측정소 목록 (공식 근접 측정소가 없을 때만 지연 호출됨)"""
    data = api_get(
        f"{AIR_BASE}/MsrstnInfoInqireSvc/getMsrstnList",
        params={"serviceKey": AIR_KEY, "returnType": "json", "numOfRows": 1000, "pageNo": 1},
        timeout=(3, 8),
    )
    items = data.get("response", {}).get("body", {}).get("items") or []
    df = pd.DataFrame(items)
    if df.empty or not {"stationName", "dmX", "dmY"}.issubset(df.columns):
        return pd.DataFrame()

    df["lat"] = pd.to_numeric(df["dmX"], errors="coerce")
    df["lon"] = pd.to_numeric(df["dmY"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        return pd.DataFrame()
    if df["lat"].median() > 60:
        df["lat"], df["lon"] = df["lon"], df["lat"]
    if "addr" not in df.columns:
        df["addr"] = None
    return df[["stationName", "addr", "lat", "lon"]].reset_index(drop=True)


def find_nearest_stations(lat, lon, k=5):
    """공식 근접 측정소 API가 비어 있는 곳을 위해 직접 계산한 가까운 측정소 후보"""
    stations = get_all_stations()
    if stations.empty:
        return []

    p1 = np.radians(lat)
    p2 = np.radians(stations["lat"].to_numpy())
    dphi = p2 - p1
    dlmb = np.radians(stations["lon"].to_numpy() - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))

    result = []
    seen = set()
    for i in np.argsort(dist)[:k]:
        if dist[i] > MAX_APPROX_KM:
            break
        row = stations.iloc[int(i)]
        if row["stationName"] in seen:
            continue
        seen.add(row["stationName"])
        result.append(
            {
                "name": row["stationName"],
                "distance": round(float(dist[i]), 1),
                "addr": row["addr"],
                "pos": (float(row["lat"]), float(row["lon"])),
                "approx": True,
            }
        )
    return result


def has_valid_reading(items):
    """측정소 응답에 숫자로 된 PM10/PM2.5 값이 하나라도 있는지 확인"""
    for item in items or []:
        if not isinstance(item, dict):
            continue
        for key in ("pm10Value", "pm25Value"):
            try:
                float(item.get(key))
                return True
            except (TypeError, ValueError):
                continue
    return False


@st.cache_data(ttl=600, show_spinner=False)
def get_station_air_quality(station_name):
    """측정소 실시간 데이터 조회"""
    url = f"{AIR_BASE}/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        "serviceKey": AIR_KEY,
        "returnType": "json",
        "numOfRows": 168,
        "stationName": station_name,
        "dataTerm": "MONTH",
        "ver": "1.0",
    }
    try:
        try:
            data = api_get(url, params=params, retries=0, timeout=(3, 5))
        except requests.Timeout:
            params.update({"dataTerm": "DAILY", "numOfRows": 24})
            data = api_get(url, params=params, retries=1, timeout=(3, 5))
        items = data.get("response", {}).get("body", {}).get("items")
        return {"items": items, "fetched_at": now_kst(), "error": None}
    except Exception as e:
        logger.exception("측정소 데이터 조회 실패: %s", station_name)
        return {"items": None, "fetched_at": now_kst(), "error": describe_error(e)}


def pick_valid_station(candidates, skipped):
    """후보 목록에서 정상 데이터가 나오는 첫 측정소를 찾아 반환"""
    for cand in candidates:
        result = get_station_air_quality(cand["name"])
        if result.get("error"):
            skipped.append(f"{cand['name']}: {result['error']}")
            continue
        if has_valid_reading(result["items"]):
            return cand, result
        skipped.append(f"{cand['name']}: 데이터 없음")
    return None, None


def load_sido():
    try:
        return get_sido_air_quality()
    except Exception:
        logger.exception("시도별 대기질 조회 실패")
        return None


def get_sido_avg(sido_df, sido_key):
    """시도 약칭과 정확히 일치하는 평균값 반환"""
    if sido_df is None or sido_df.empty or not sido_key:
        return None
    sub = sido_df[sido_df["sidoName"] == sido_key]
    if sub.empty:
        return None
    return float(sub["pm10Value"].mean())


# --- UI 컴포넌트 ---
def render_freshness(data_time, fetched_at, show_measured=True):
    """데이터 신선도 표시 + 새로고침 버튼 + 지연 경고"""
    c_text, c_btn = st.columns([5, 1])
    with c_text:
        parts = []
        if show_measured and data_time is not None:
            parts.append(f"🕒 측정 {data_time.strftime('%m/%d %H:%M')} ({ago_text(data_time)})")
        parts.append(f"🔄 조회 {ago_text(fetched_at)}")
        st.caption(" · ".join(parts))
    with c_btn:
        if st.button("새로고침", use_container_width=True):
            get_sido_air_quality.clear()
            get_station_air_quality.clear()
            st.rerun()
    if data_time is not None and minutes_since(data_time) > STALE_MINUTES:
        st.warning("⚠️ 측정값 업데이트가 지연되고 있어요. 측정소 점검이나 통신 문제일 수 있습니다.")


def metric_card(label, value, meta=None, caption=None):
    with st.container(border=True):
        st.metric(label, value)
        if meta is not None:
            st.markdown(badge(meta), unsafe_allow_html=True)
        if caption:
            st.caption(caption)


def region_card(rank, row):
    meta = get_grade(row["pm10Value"], "PM10")
    with st.container(border=True):
        left, right = st.columns([1, 1.4])
        left.markdown(f"**{rank}. {row['sidoName']}**")
        right.markdown(
            f"**{row['pm10Value']:.1f}** ㎍/㎥ &nbsp; {badge(meta)}",
            unsafe_allow_html=True,
        )


def fit_view(points):
    """(위도, 경도) 목록이 한 화면에 들어오도록 중심/줌 계산"""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    if len(points) == 1:
        return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)
    span = max(
        max(lats) - min(lats),
        (max(lons) - min(lons)) * math.cos(math.radians(center_lat)),
    )
    zoom = min(13.5, max(8.0, math.log2(360 / (max(span, 0.001) * 2.2))))
    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)


def render_map(x, y, place_name, station_name=None, station_pos=None, rgb=(128, 128, 128)):
    points = [
        {"lat": y, "lon": x, "name": place_name, "kind": "검색 위치",
         "color": [38, 50, 56, 230], "radius": 150}
    ]
    layers = []
    view_points = [(y, x)]

    if station_pos:
        s_lat, s_lon = station_pos
        points.append(
            {"lat": s_lat, "lon": s_lon, "name": station_name, "kind": "측정소",
             "color": list(rgb) + [230], "radius": 250}
        )
        view_points.append((s_lat, s_lon))
        line_df = pd.DataFrame([{"sx": x, "sy": y, "tx": s_lon, "ty": s_lat}])
        layers.append(
            pdk.Layer(
                "LineLayer",
                line_df,
                get_source_position="[sx, sy]",
                get_target_position="[tx, ty]",
                get_color=[120, 120, 120, 200],
                get_width=2,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            pd.DataFrame(points),
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=7,
            radius_max_pixels=18,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=fit_view(view_points),
        map_style="road",
        tooltip={"html": "<b>{name}</b><br/>{kind}"},
    )
    try:
        st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)
    if station_pos:
        st.caption("⚫ 검색 위치 · ● 측정소 (색 = PM10 등급) · 마커에 마우스를 올리면 이름이 보여요")
    else:
        st.caption("⚫ 검색 위치")


def build_trend_chart(series, kind):
    """시간별 꺾은선 + 등급 구간 배경띠"""
    limits = THRESHOLDS[kind]
    y_top = max(float(series["value"].max()) * 1.15, limits[1] * 1.3)
    y_scale = alt.Scale(domain=[0, y_top], nice=False)

    lows = [0, *limits]
    highs = [*limits, y_top]
    band_rows = [
        {"등급": meta[0], "lo": lo, "hi": min(hi, y_top)}
        for meta, lo, hi in zip(GRADE_META, lows, highs)
        if lo < y_top
    ]
    grade_scale = alt.Scale(
        domain=[m[0] for m in GRADE_META], range=[m[2] for m in GRADE_META]
    )

    bands = (
        alt.Chart(pd.DataFrame(band_rows))
        .mark_rect(opacity=0.18)
        .encode(
            y=alt.Y("lo:Q", scale=y_scale, title="㎍/㎥"),
            y2="hi:Q",
            color=alt.Color(
                "등급:N",
                scale=grade_scale,
                legend=alt.Legend(title="등급", orient="bottom", direction="horizontal"),
            ),
        )
    )

    line = (
        alt.Chart(series)
        .mark_line(
            color="#5C6BC0",
            point=alt.OverlayMarkDef(color="#5C6BC0", size=25),
        )
        .encode(
            x=alt.X("time:T", title="시간", axis=alt.Axis(format="%m/%d %H시", labelAngle=-30)),
            y=alt.Y("value:Q", scale=y_scale, title="㎍/㎥"),
            tooltip=[
                alt.Tooltip("time:T", title="시각", format="%m/%d %H:%M"),
                alt.Tooltip("value:Q", title=kind, format=".0f"),
            ],
        )
    )
    return (bands + line).properties(height=280)


# --- Streamlit UI 구성 ---
st.title("☁️ 실시간 대기질 모니터링")

query = st.text_input(
    "🔍 지역명 또는 장소명을 입력하세요",
    placeholder="도시, 주소, 장소명(예: 수원대, 인천국제공항)을 자유롭게 입력해보세요",
).strip()

# 1. 검색어가 없을 때: 메인 화면 (Best / Worst)
if not query:
    with st.spinner("전국 대기질 요약 정보를 불러오고 있어요..."):
        sido = load_sido()
    sido_df = sido["df"] if sido else pd.DataFrame()

    if not sido_df.empty:
        st.markdown("---")
        st.markdown("#### 🏆 현재 전국 미세먼지 (PM10) 요약")

        n = min(3, len(sido_df) // 2) or 1
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### ✨ 공기가 맑은 곳 (Best {n})")
            for rank, (_, row) in enumerate(sido_df.head(n).iterrows(), start=1):
                region_card(rank, row)
        with col2:
            st.markdown(f"##### ⚠️ 주의가 필요한 곳 (Worst {n})")
            for rank, (_, row) in enumerate(sido_df.tail(n).iloc[::-1].iterrows(), start=1):
                region_card(rank, row)

        render_freshness(sido["data_time"], sido["fetched_at"])
        render_legend()
    else:
        st.warning("전국 대기질 평균 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

# 2. 검색어가 있을 때: 후보 선택 → 결과 화면
try:
    cands = get_kakao_candidates(query)
except Exception as e:
    logger.exception("위치 검색 실패")
    st.error(f"위치를 검색하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요. (오류 유형: {type(e).__name__})")
    st.stop()

if not cands:
    st.error("위치를 찾을 수 없습니다. 장소명이나 주소를 다시 확인해주세요.")
    st.stop()

if len(cands) > 1:
    idx = st.selectbox(
        f"'{query}' 검색 결과가 {len(cands)}개 있어요. 원하는 위치를 선택하세요",
        range(len(cands)),
        format_func=lambda i: cands[i]["label"],
        key=f"cand_{query}",
    )
else:
    idx = 0

chosen = cands[idx]
x, y, sido_key = chosen["x"], chosen["y"], chosen["sido"]
place_label = chosen["label"].removeprefix("📍 ")

# 단일 스피너 안에서 측정소 탐색 및 데이터 로드를 한 번에 처리
station, air, skipped = None, None, []
try:
    with st.spinner("대기질 데이터를 불러오고 있어요..."):
        tm = convert_to_tm(x, y)
        official = get_nearby_stations(*tm) if tm else []

        # 1차: 공식 근접 측정소에서 먼저 유효한 데이터 탐색 (빠른 처리)
        if official:
            station, air = pick_valid_station(official, skipped)

        # 2차: 공식 측정소가 없거나 모두 실패했을 때만 전국 측정소 거리 계산 수행 (지연 실행)
        if station is None:
            try:
                official_names = {c["name"] for c in official}
                extra = [
                    c for c in find_nearest_stations(y, x)
                    if c["name"] not in official_names
                ]
            except Exception:
                logger.exception("최근접 측정소 탐색 실패")
                extra = []
            if extra:
                station, air = pick_valid_station(extra, skipped)

        sido = load_sido()
except Exception as e:
    logger.exception("측정소 조회 실패")
    st.error(f"측정소를 찾는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요. (오류 유형: {type(e).__name__})")
    st.stop()

sido_df = sido["df"] if sido else pd.DataFrame()
sido_avg_val = get_sido_avg(sido_df, sido_key)


def render_sido_fallback(reason):
    """측정소 데이터를 쓸 수 없을 때 시도 평균으로 대체해서 보여줌"""
    label = sido_key or "해당 지역"
    st.warning(f"📍 '{place_label}' {reason} **{label} 전체 평균** 대기질로 대체하여 보여드립니다.")

    if sido_avg_val is not None:
        meta = get_grade(sido_avg_val, "PM10")
        st.markdown(
            f"### 현재 {label} 평균 대기 상태 (PM10 기준) &nbsp; {badge(meta, size='1.1rem')}",
            unsafe_allow_html=True,
        )
        c1, _ = st.columns([1, 2])
        with c1:
            metric_card("미세먼지 (PM10) 평균", f"{sido_avg_val:.1f} ㎍/㎥", meta)
        render_freshness(sido["data_time"], sido["fetched_at"])

        st.markdown("---")
        st.markdown("#### 🗺️ 검색 위치")
        render_map(x, y, place_label, rgb=meta[4])
        render_legend()
    else:
        st.error("현재 해당 지역의 평균 데이터를 불러올 수 없습니다.")
    st.stop()


# --- 사각지대 예외 처리 ---
if station is None:
    if not official and not skipped:
        render_sido_fallback("반경 내에 개별 측정소가 없어,")
    else:
        render_sido_fallback("근처 측정소가 점검 중이어서,")

station_name = station["name"]

df = pd.DataFrame(air["items"])
for col in ("dataTime", "pm10Value", "pm25Value"):
    if col not in df.columns:
        df[col] = None

df["time"] = parse_data_time(df["dataTime"])
df["pm10"] = pd.to_numeric(df["pm10Value"], errors="coerce")
df["pm25"] = pd.to_numeric(df["pm25Value"], errors="coerce")
df = df.sort_values("time", ascending=False).reset_index(drop=True)

valid = df.dropna(subset=["pm10"])
latest = valid.iloc[0] if not valid.empty else df.iloc[0]

pm10_val = None if pd.isna(latest["pm10"]) else float(latest["pm10"])
pm25_val = None if pd.isna(latest["pm25"]) else float(latest["pm25"])
pm10_meta = get_grade(pm10_val, "PM10")
pm25_meta = get_grade(pm25_val, "PM2.5")
measured_at = latest["time"] if pd.notna(latest["time"]) else None

distance = station.get("distance")
dist_text = f" · 약 {distance}km" if distance not in (None, "") else ""

# --- (요구사항 1) 일반 사용자를 위한 깔끔한 안내 문구 ---
if station.get("approx"):
    st.info(
        f"📍 '{place_label}'에서 가장 인접한 **{station_name}**{dist_text} 측정소 기준으로 안내해 드립니다."
    )
elif skipped:
    st.info(
        f"📍 가장 인접한 측정소가 점검 중이어서, 다음으로 가까운 **{station_name}**{dist_text} 측정소 기준으로 안내해 드립니다."
    )
else:
    st.success(f"📍 가장 가까운 측정소 · **{station_name}**{dist_text}")
st.caption(f"검색 위치: {place_label}")

st.markdown(
    f"### 현재 대기 상태 (PM10 기준) &nbsp; {badge(pm10_meta, size='1.1rem')}",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    metric_card(
        "미세먼지 (PM10)",
        f"{pm10_val:.0f} ㎍/㎥" if pm10_val is not None else "점검 중",
        pm10_meta,
    )
with c2:
    metric_card(
        "초미세먼지 (PM2.5)",
        f"{pm25_val:.0f} ㎍/㎥" if pm25_val is not None else "점검 중",
        pm25_meta,
    )
with c3:
    metric_card(
        "최근 측정 시간",
        measured_at.strftime("%m/%d %H:%M") if measured_at is not None else "-",
        caption=ago_text(measured_at),
    )

render_freshness(measured_at, air["fetched_at"], show_measured=False)

st.markdown("---")

# --- 지도 + 지역 평균 비교 ---
col_map, col_cmp = st.columns([1.2, 1])

with col_map:
    st.markdown("#### 🗺️ 검색 위치와 측정소")
    station_pos = station.get("pos")
    if station_pos is None:
        try:
            station_pos = get_station_location(station_name, station.get("addr"))
        except Exception:
            logger.exception("측정소 위치 조회 실패")
            station_pos = None
    render_map(x, y, place_label, station_name, station_pos, rgb=pm10_meta[4])

with col_cmp:
    st.markdown("#### 📊 지역 평균 비교")
    if pm10_val is None:
        st.warning("현재 측정소가 점검 중이어서 비교 불가합니다.")
    elif sido_avg_val is None:
        st.info("해당 시도의 평균 데이터를 불러오지 못했습니다.")
    else:
        comp = pd.DataFrame(
            {
                "구분": ["현재 측정소", f"{sido_key} 평균"],
                "PM10": [pm10_val, sido_avg_val],
            }
        )
        base = alt.Chart(comp).encode(
            y=alt.Y("구분:N", sort=None, title=""),
            x=alt.X("PM10:Q", title="수치 (㎍/㎥)"),
        )
        bars = base.mark_bar(size=44).encode(
            color=alt.condition(
                alt.datum["구분"] == "현재 측정소",
                alt.value(pm10_meta[2]),
                alt.value("#B0BEC5"),
            )
        )
        labels = base.mark_text(
            align="left", baseline="middle", dx=4, fontSize=14, fontWeight="bold"
        ).encode(text=alt.Text("PM10:Q", format=".1f"))
        st.altair_chart((bars + labels).properties(height=190), use_container_width=True)

        diff = pm10_val - sido_avg_val
        pct = diff / sido_avg_val * 100 if sido_avg_val else 0
        with st.container(border=True):
            st.metric(
                f"{sido_key} 평균 대비",
                f"{diff:+.1f} ㎍/㎥",
                f"{pct:+.0f}%",
                delta_color="inverse",
            )

    with st.container(border=True):
        st.markdown("**🏢 측정소 정보**")
        st.markdown(
            f"{station_name}  \n{station.get('addr') or '-'}"
            + (f"  \n검색 위치에서 약 {distance}km" if distance not in (None, "") else "")
        )

st.markdown("---")

# --- 시간별 추이 ---
st.markdown("#### 📈 시간별 추이")
opt1, opt2 = st.columns(2)
kind = opt1.radio("항목", ["PM10", "PM2.5"], horizontal=True, key="trend_kind")
period = opt2.radio("기간", ["24시간", "3일", "7일"], horizontal=True, key="trend_period")
hours = {"24시간": 24, "3일": 72, "7일": 168}[period]

value_col = "pm10" if kind == "PM10" else "pm25"
series = df.dropna(subset=["time"])[["time", value_col]].rename(columns={value_col: "value"})
if not series.empty:
    series = series[series["time"] >= series["time"].max() - pd.Timedelta(hours=hours)]
    series = series.sort_values("time")

if series.empty or series["value"].notna().sum() == 0:
    st.info("차트를 그릴 수 있는 데이터가 부족합니다.")
else:
    data_span_h = (df["time"].max() - df["time"].min()).total_seconds() / 3600
    if hours > data_span_h + 2:
        st.caption(f"ℹ️ 이 측정소는 최근 약 {int(data_span_h)}시간 데이터만 불러와져서 그 범위만 표시돼요.")
    st.altair_chart(build_trend_chart(series, kind), use_container_width=True)

render_legend()