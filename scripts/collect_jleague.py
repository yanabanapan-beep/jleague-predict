"""
Jリーグ データ収集スクリプト
------------------------------------
データソース: Yahoo!スポーツ (https://soccer.yahoo.co.jp/jleague/category/j1/schedule)
  - Jリーグの公式無料APIが存在しない(API-Footballの無料プランは現在シーズンのデータに
    対応していないことが判明したため)、公開されている試合日程ページをスクレイピングして
    データを取得する方式にしている。
  - 個人利用の範囲で、短時間に大量アクセスしないよう配慮すること。
  - サイトのHTML構造が変わると動かなくなる可能性がある(non-API方式の宿命)。

SEASON_ID について:
  - Jリーグ戦(節)ごとのURLには "シーズンID" という数字が含まれており、
    シーズン(年度・ステージ)が変わるとこの数字も変わる。
  - 確認方法:
    1. https://soccer.yahoo.co.jp/jleague/category/j1/schedule をブラウザで開く
    2. 節を切り替えるリンクをクリックすると URL が
       https://soccer.yahoo.co.jp/jleague/category/j1/schedule/【ここの数字】/【節番号】/
       のようになるので、その数字を下の SEASON_ID に設定する

使い方:
  1. pip install -r requirements.txt
  2. python scripts/collect_jleague.py を実行
"""

import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

SEASON_ID = 31467  # 2026/27シーズン J1リーグ戦。シーズンが変わったら要確認(上記コメント参照)
BASE_URL = "https://soccer.yahoo.co.jp/jleague/category/j1/schedule"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; jleague-predict-dashboard/1.0)"}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jleague")
os.makedirs(DATA_DIR, exist_ok=True)

_RESULT_COLUMNS = ["fixture_id", "round", "date", "home_team", "away_team", "home_goals", "away_goals", "status"]


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


def _fetch_round(round_no: int = None) -> pd.DataFrame:
    """
    指定した節(ラウンド)の対戦カード一覧を取得する。
    round_no を指定しない場合、サイト側で選ばれている既定の節が表示される。
    """
    url = f"{BASE_URL}/{SEASON_ID}/{round_no}/" if round_no else BASE_URL
    res = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

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


def _current_round_number() -> int:
    """既定表示されている節の番号を読み取る(例: "第4節" -> 4)。"""
    default_df = _fetch_round(None)
    if default_df.empty:
        raise RuntimeError("節の情報を取得できませんでした。サイト構造が変わった可能性があります。")
    label = default_df.iloc[0]["round"] or ""
    match = re.search(r"(\d+)", label)
    if not match:
        raise RuntimeError(f"節番号を読み取れませんでした(取得した表示: {label!r})")
    return int(match.group(1))


def fetch_recent_results(last_n: int = 30) -> pd.DataFrame:
    """
    直近の消化済み試合結果を、節を過去にさかのぼりながら last_n 件集める。
    """
    current_round = _current_round_number()
    collected = []
    round_no = current_round
    while round_no >= 1 and sum(len(d) for d in collected) < last_n:
        df = _fetch_round(round_no)
        finished = df[df["home_goals"].notna()]
        if not finished.empty:
            collected.append(finished)
        round_no -= 1
        time.sleep(0.3)  # 短時間に連続アクセスしすぎないよう配慮

    if not collected:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    result = pd.concat(collected, ignore_index=True)
    return result.head(last_n).reset_index(drop=True)


def fetch_upcoming_fixtures(next_n: int = 10) -> pd.DataFrame:
    """
    今後の対戦カード(未消化試合)を、節を進めながら next_n 件集める。
    """
    current_round = _current_round_number()
    collected = []
    round_no = current_round
    max_round = current_round + 10  # 安全のため探索範囲に上限を設ける
    while round_no <= max_round and sum(len(d) for d in collected) < next_n:
        df = _fetch_round(round_no)
        upcoming = df[df["home_goals"].isna()]
        if not upcoming.empty:
            collected.append(upcoming)
        round_no += 1
        time.sleep(0.3)

    if not collected:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    result = pd.concat(collected, ignore_index=True)
    return result.head(next_n).reset_index(drop=True)


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
    print("直近の試合結果を取得中...")
    results_df = fetch_recent_results(last_n=30)
    results_path = os.path.join(DATA_DIR, f"results_{datetime.now().date()}.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"保存しました: {results_path}")

    print("今後の対戦カードを取得中...")
    upcoming_df = fetch_upcoming_fixtures(next_n=10)
    upcoming_path = os.path.join(DATA_DIR, f"upcoming_{datetime.now().date()}.csv")
    upcoming_df.to_csv(upcoming_path, index=False, encoding="utf-8-sig")
    print(f"保存しました: {upcoming_path}")

    print("チームごとの直近成績を集計中...")
    form_df = build_team_form(results_df)
    form_path = os.path.join(DATA_DIR, f"team_form_{datetime.now().date()}.csv")
    form_df.to_csv(form_path, index=False, encoding="utf-8-sig")
    print(f"保存しました: {form_path}")

    print("\n=== チーム成績サマリー(上位5) ===")
    print(form_df.head())


if __name__ == "__main__":
    main()
