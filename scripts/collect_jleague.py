"""
Jリーグ データ収集スクリプト
------------------------------------
データソース: Yahoo!スポーツ (https://soccer.yahoo.co.jp/jleague/category/{j1|j2}/schedule)
  - Jリーグの公式無料APIが存在しない(API-Footballの無料プランは現在シーズンのデータに
    対応していないことが判明したため)、公開されている試合日程ページをスクレイピングして
    データを取得する方式にしている。
  - 個人利用の範囲で、短時間に大量アクセスしないよう配慮すること。
  - サイトのHTML構造が変わると動かなくなる可能性がある(non-API方式の宿命)。

シーズンID・節数について:
  - 節を切り替えるリンクがページに埋め込まれているので、そこから
    シーズンID(年度・大会ごとに変わる数字)と全体の節数を自動検出している。
  - そのため、シーズンが変わっても基本的にコードの変更なしで動く想定。

使い方:
  1. pip install -r requirements.txt
  2. python scripts/collect_jleague.py を実行(J1・J2 両方のデータを取得する)
"""

import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

CATEGORIES = {"j1": "J1", "j2": "J2"}  # Yahoo!スポーツ側のカテゴリ名 -> 表示名

BASE_URL_TEMPLATE = "https://soccer.yahoo.co.jp/jleague/category/{category}/schedule"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; jleague-predict-dashboard/1.0)"}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jleague")
os.makedirs(DATA_DIR, exist_ok=True)

_RESULT_COLUMNS = ["fixture_id", "round", "date", "home_team", "away_team", "home_goals", "away_goals", "status"]

_season_cache = {}  # category -> {"season_id", "max_round", "current_round", "default_df"}


def _team_name(team_td) -> str:
    """チーム名のtd要素から、チーム名テキストを取り出す。"""
    for span in team_td.select("a.sc-tableGame__team span"):
        text = span.get_text(strip=True)
        if text:
            return text
    return ""


def _parse_score(score_td):
    """
    スコアのtd要素から (home_goals, away_goals, status, fixture_id) を取り出す。
    未消化の試合はスコアが "-" と表示されるので、その場合は goals は None になる。
    """
    if score_td is None:
        return None, None, None, None

    link = score_td.select_one("a.sc-tableGame__score")
    fixture_id = None
    if link and link.get("href"):
        match = re.search(r"/game/(\d+)", link["href"])
        if match:
            fixture_id = match.group(1)

    detail = score_td.select_one("p.sc-tableGame__scoreDetail")
    status_p = score_td.select_one("p.sc-tableGame__status")
    status = status_p.get_text(strip=True) if status_p else None

    home_goals, away_goals = None, None
    if detail:
        text = detail.get_text(" ", strip=True)
        parts = [p.strip() for p in text.split("-")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            home_goals, away_goals = int(parts[0]), int(parts[1])

    return home_goals, away_goals, status, fixture_id


def _parse_round_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    for tr in soup.select("tr"):
        date_td = tr.select_one("td.sc-tableGame__data--date")
        team_tds = tr.select("td.sc-tableGame__data--team")
        if date_td is None or len(team_tds) < 2:
            continue

        category_td = tr.select_one("td.sc-tableGame__data--category")
        round_label = category_td.get_text(strip=True) if category_td else None

        score_td = tr.select_one("td.sc-tableGame__data--score")
        home_goals, away_goals, status, fixture_id = _parse_score(score_td)

        rows.append({
            "fixture_id": fixture_id,
            "round": round_label,
            "date": date_td.get_text(" ", strip=True),
            "home_team": _team_name(team_tds[0]),
            "away_team": _team_name(team_tds[1]),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "status": status,
        })

    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


def _fetch_html(url: str) -> str:
    res = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    res.raise_for_status()
    return res.text


def _discover_season(category: str) -> dict:
    """
    既定(節を指定しない)の日程ページから、シーズンID・現在の節番号・全体の節数を検出する。
    同じcategoryに対する2回目以降の呼び出しはキャッシュを返す。
    """
    if category in _season_cache:
        return _season_cache[category]

    html = _fetch_html(BASE_URL_TEMPLATE.format(category=category))

    ids = re.findall(rf"category/{category}/schedule/(\d+)/(\d+)", html)
    if not ids:
        raise RuntimeError(
            f"({CATEGORIES.get(category, category)}) シーズン情報を検出できませんでした。"
            "Yahoo!スポーツのページ構造が変わった可能性があります。"
        )
    season_id = int(ids[0][0])
    max_round = max(int(r) for _, r in ids)

    default_df = _parse_round_table(html)
    if default_df.empty:
        raise RuntimeError(f"({CATEGORIES.get(category, category)}) 節情報を取得できませんでした。")
    label = default_df.iloc[0]["round"] or ""
    match = re.search(r"(\d+)", label)
    if not match:
        raise RuntimeError(f"({CATEGORIES.get(category, category)}) 節番号を読み取れませんでした(表示: {label!r})")
    current_round = int(match.group(1))

    info = {
        "season_id": season_id,
        "max_round": max_round,
        "current_round": current_round,
        "default_df": default_df,
    }
    _season_cache[category] = info
    return info


def _fetch_round(category: str, round_no: int) -> pd.DataFrame:
    info = _discover_season(category)
    if round_no == info["current_round"]:
        return info["default_df"]  # 既定ページを再利用し、無駄なリクエストを避ける
    url = f"{BASE_URL_TEMPLATE.format(category=category)}/{info['season_id']}/{round_no}/"
    return _parse_round_table(_fetch_html(url))


def fetch_recent_results(category: str = "j1", last_n: int = None) -> pd.DataFrame:
    """
    消化済み試合結果を取得する。
    last_n=None の場合、今シーズンの第1節から現在までの全試合を取得する。
    last_n を指定した場合は、直近の指定件数だけを節をさかのぼって集める。
    """
    info = _discover_season(category)
    collected = []

    for round_no in range(info["current_round"], 0, -1):
        df = _fetch_round(category, round_no)
        finished = df[df["home_goals"].notna()]
        if not finished.empty:
            collected.append(finished)
        if last_n is not None and sum(len(d) for d in collected) >= last_n:
            break
        time.sleep(0.3)  # 短時間に連続アクセスしすぎないよう配慮

    if not collected:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    result = pd.concat(collected, ignore_index=True)
    if last_n is not None:
        result = result.head(last_n)
    return result.reset_index(drop=True)


def fetch_upcoming_fixtures(category: str = "j1", next_n: int = 10) -> pd.DataFrame:
    """
    今後の対戦カード(未消化試合)を取得する。
    next_n=None の場合、今シーズン残り全ての対戦カードを取得する。
    """
    info = _discover_season(category)
    collected = []

    for round_no in range(info["current_round"], info["max_round"] + 1):
        df = _fetch_round(category, round_no)
        upcoming = df[df["home_goals"].isna()]
        if not upcoming.empty:
            collected.append(upcoming)
        if next_n is not None and sum(len(d) for d in collected) >= next_n:
            break
        time.sleep(0.3)

    if not collected:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    result = pd.concat(collected, ignore_index=True)
    if next_n is not None:
        result = result.head(next_n)
    return result.reset_index(drop=True)


def build_team_form(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    チームごとの直近成績(得点・失点・勝敗数)を集計した簡易特徴量。
    予測モデルの入力特徴量として使う。
    """
    records = []
    if not results_df.empty:
        teams = pd.unique(results_df[["home_team", "away_team"]].values.ravel())
    else:
        teams = []

    for team in teams:
        home_games = results_df[results_df["home_team"] == team]
        away_games = results_df[results_df["away_team"] == team]

        goals_for = home_games["home_goals"].sum() + away_games["away_goals"].sum()
        goals_against = home_games["away_goals"].sum() + away_games["home_goals"].sum()
        wins = (
            (home_games["home_goals"] > home_games["away_goals"]).sum()
            + (away_games["away_goals"] > away_games["home_goals"]).sum()
        )
        games_played = len(home_games) + len(away_games)

        records.append({
            "team": team,
            "games_played": games_played,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_diff": goals_for - goals_against,
            "wins": wins,
        })

    columns = ["team", "games_played", "goals_for", "goals_against", "goal_diff", "wins"]
    return pd.DataFrame(records, columns=columns).sort_values("goal_diff", ascending=False)


def main():
    for category, label in CATEGORIES.items():
        print(f"\n===== {label} =====")

        print("今シーズンの全試合結果を取得中...")
        results_df = fetch_recent_results(category=category, last_n=None)
        results_path = os.path.join(DATA_DIR, f"results_{category}_{datetime.now().date()}.csv")
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {results_path} ({len(results_df)}試合)")

        print("今後の対戦カードを取得中...")
        upcoming_df = fetch_upcoming_fixtures(category=category, next_n=10)
        upcoming_path = os.path.join(DATA_DIR, f"upcoming_{category}_{datetime.now().date()}.csv")
        upcoming_df.to_csv(upcoming_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {upcoming_path} ({len(upcoming_df)}試合)")

        print("チームごとの成績を集計中...")
        form_df = build_team_form(results_df)
        form_path = os.path.join(DATA_DIR, f"team_form_{category}_{datetime.now().date()}.csv")
        form_df.to_csv(form_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {form_path}")

        print(f"\n=== {label} チーム成績サマリー(上位5) ===")
        print(form_df.head())


if __name__ == "__main__":
    main()
