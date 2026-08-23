"""
Jリーグ データ収集スクリプト
------------------------------------
データソース: API-Football (https://www.api-football.com/)
  - 無料アカウント登録で API KEY を取得できます(無料枠: 1日100リクエストまで)
  - 登録: https://dashboard.api-football.com/register
  - 登録後、ダッシュボードで API KEY をコピーして、環境変数 API_FOOTBALL_KEY に設定してください

使い方:
  1. pip install -r requirements.txt
  2. 環境変数に API キーを設定する
     Windows(PowerShell)の例:
       $env:API_FOOTBALL_KEY = "あなたのAPIキー"
  3. python scripts/collect_jleague.py を実行

このスクリプトがすること:
  - Jリーグ(J1)の直近の試合結果を取得
  - 出場予定の今後の試合(対戦カード)を取得
  - チームごとの直近成績(勝敗・得失点)をCSVに集計・保存
  - 出場停止・負傷情報を取得

TODO(次のステップ):
  - xG(期待得点)は無料プランでは取得不可なことが多いので、別ソースを検討
  - オッズ比較をしたい場合は odds エンドポイントを追加(有料プラン必要な場合あり)
"""

import os
from datetime import datetime

import pandas as pd
import requests

# ========= 設定 =========
BASE_URL = "https://v3.football.api-sports.io"

# Jリーグ(J1)のリーグID(API-Football上のID。念のため実行時に確認する関数を用意)
J1_LEAGUE_ID = 98  # ※ズレている場合は find_league_id() で確認してください

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jleague")
os.makedirs(DATA_DIR, exist_ok=True)


def _headers() -> dict:
    """
    APIキーはリクエストの都度、環境変数から読み直す。
    (Streamlit Cloudの Secrets 経由で実行中に設定されるケースがあるため、
     モジュール読み込み時の一度きりの読み取りにはしていない)
    """
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        raise RuntimeError(
            "API_FOOTBALL_KEY が設定されていません。"
            "環境変数(またはStreamlit CloudのSecrets)に API-Football の API キーを設定してください。"
        )
    return {"x-apisports-key": api_key}


def find_league_id():
    """
    Jリーグの正確なリーグIDを確認するための補助関数。
    APIの仕様変更でIDがズレることがあるので、迷ったら実行してみてください。
    """
    headers = _headers()
    url = f"{BASE_URL}/leagues"
    params = {"search": "J1 League"}
    res = requests.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()
    for item in data.get("response", []):
        league = item["league"]
        print(f"ID: {league['id']}  名前: {league['name']}  国: {item['country']['name']}")


def fetch_recent_results(season: int, last_n: int = 20) -> pd.DataFrame:
    """
    直近の試合結果を取得してDataFrameで返す。
    season: 例 2026
    last_n: 直近何試合分を取得するか
    """
    headers = _headers()
    url = f"{BASE_URL}/fixtures"
    params = {
        "league": J1_LEAGUE_ID,
        "season": season,
        "last": last_n,
    }
    res = requests.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()

    rows = []
    for item in data.get("response", []):
        fixture = item["fixture"]
        teams = item["teams"]
        goals = item["goals"]
        rows.append({
            "fixture_id": fixture["id"],
            "date": fixture["date"],
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
            "home_goals": goals["home"],
            "away_goals": goals["away"],
            "home_win": teams["home"]["winner"],
            "away_win": teams["away"]["winner"],
        })
    columns = ["fixture_id", "date", "home_team", "away_team", "home_goals", "away_goals", "home_win", "away_win"]
    return pd.DataFrame(rows, columns=columns)


def fetch_upcoming_fixtures(season: int, next_n: int = 10) -> pd.DataFrame:
    """
    今後の対戦カード(未消化試合)を取得する。
    """
    headers = _headers()
    url = f"{BASE_URL}/fixtures"
    params = {
        "league": J1_LEAGUE_ID,
        "season": season,
        "next": next_n,
    }
    res = requests.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()

    rows = []
    for item in data.get("response", []):
        fixture = item["fixture"]
        teams = item["teams"]
        rows.append({
            "fixture_id": fixture["id"],
            "date": fixture["date"],
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
        })
    columns = ["fixture_id", "date", "home_team", "away_team"]
    return pd.DataFrame(rows, columns=columns)


def fetch_injuries(season: int, team_ids: list = None) -> pd.DataFrame:
    """
    出場停止・負傷情報を取得する。
    team_ids を指定しない場合はリーグ全体の直近の負傷情報を取得する。
    (API-Footballの無料枠だとリクエスト数を消費しやすいので、
     必要なチームだけ絞って取得するのがおすすめ)
    """
    headers = _headers()
    url = f"{BASE_URL}/injuries"
    rows = []

    targets = team_ids if team_ids else [None]

    for team_id in targets:
        params = {"league": J1_LEAGUE_ID, "season": season}
        if team_id:
            params["team"] = team_id

        res = requests.get(url, headers=headers, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        for item in data.get("response", []):
            player = item["player"]
            team = item["team"]
            fixture = item.get("fixture", {})
            rows.append({
                "team": team["name"],
                "player": player["name"],
                "type": player.get("type"),      # 例: Missing Fixture
                "reason": player.get("reason"),  # 例: Injury, Suspended
                "fixture_id": fixture.get("id"),
                "fixture_date": fixture.get("date"),
            })

    columns = ["team", "player", "type", "reason", "fixture_id", "fixture_date"]
    return pd.DataFrame(rows, columns=columns)


def build_team_form(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    チームごとの直近成績(得点・失点・勝敗数)を集計した簡易特徴量。
    予測モデルの入力特徴量として使う。
    """
    records = []
    teams = pd.unique(results_df[["home_team", "away_team"]].values.ravel())

    for team in teams:
        home_games = results_df[results_df["home_team"] == team]
        away_games = results_df[results_df["away_team"] == team]

        goals_for = home_games["home_goals"].sum() + away_games["away_goals"].sum()
        goals_against = home_games["away_goals"].sum() + away_games["home_goals"].sum()
        wins = home_games["home_win"].sum() + away_games["away_win"].sum()
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
    season = 2026  # 必要に応じて変更

    print("直近の試合結果を取得中...")
    results_df = fetch_recent_results(season=season, last_n=30)
    results_path = os.path.join(DATA_DIR, f"results_{datetime.now().date()}.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"保存しました: {results_path}")

    print("今後の対戦カードを取得中...")
    upcoming_df = fetch_upcoming_fixtures(season=season, next_n=10)
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

    print("\n出場停止・負傷情報を取得中...")
    injuries_df = fetch_injuries(season=season)
    injuries_path = os.path.join(DATA_DIR, f"injuries_{datetime.now().date()}.csv")
    injuries_df.to_csv(injuries_path, index=False, encoding="utf-8-sig")
    print(f"保存しました: {injuries_path}")


if __name__ == "__main__":
    # 初回だけ実行してリーグIDを確認したい場合はコメントアウトを外す
    # find_league_id()

    main()
