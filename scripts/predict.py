"""
Jリーグ 試合結果予測スクリプト
------------------------------------
collect_jleague.py が出力した「消化済み試合結果(results)」と「今後の対戦カード(upcoming)」
の最新CSVを読み込み、各対戦カードの勝敗予測(勝率%)とチームのELOレーティングを計算して
CSVに保存する。

採用している手法(いずれもサッカー予想で伝統的に使われる統計手法):

1. ポアソン分布モデル(勝率計算に使用)
   各チームの「平均得点力」「平均失点力」をリーグ平均との比率で表し、
   対戦カードごとに期待得点(平均何点入りそうか)を算出。
   その期待得点をポアソン分布に当てはめて、有り得る全スコアの組み合わせから
   ホーム勝ち/引き分け/アウェイ勝ちの確率を積み上げて計算する。

2. ELOレーティング(チームの強さの指標として表示に使用)
   対戦結果を古い順に処理し、勝てば相手の強さに応じてレーティングが上がり、
   負ければ下がる仕組み。チェスのレーティングと同じ考え方。
   「どんな相手に勝ったか」を考慮できるため、単純な得失点差より
   相手の強さを反映した指標になる。

使い方:
  python scripts/predict.py
"""

import glob
import math
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd

import collect_jleague

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jleague")
CATEGORIES = collect_jleague.CATEGORIES  # {"j1": "J1", "j2": "J2"}

# ===== ELOレーティングの設定 =====
ELO_INITIAL = 1500       # 全チームの初期レーティング
ELO_K = 20                # 1試合あたりのレーティング変動の大きさ(大きいほど直近の結果を重視する)
ELO_HOME_ADVANTAGE = 100  # ホームチームに上乗せする補正値

# ===== ポアソンモデルの設定 =====
POISSON_MAX_GOALS = 8     # 何点まで想定してスコアの組み合わせを計算するか(通常はこれで十分)
POISSON_PRIOR_GAMES = 4   # 試合数が少ないチームの数値をリーグ平均に寄せるための仮想試合数
                           # (シーズン序盤は数試合しかないため、そのままだと極端な予測になりやすい)


def _latest_file(prefix: str, category: str) -> str:
    pattern = os.path.join(DATA_DIR, f"{prefix}_{category}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"{pattern} が見つかりません。先に collect_jleague.py を実行してください。"
        )
    return files[-1]


def compute_elo_ratings(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    消化済み試合結果を節(ラウンド)の古い順に処理し、チームごとのELOレーティングを算出する。
    同じ節の試合は同時に行われたものとして扱い、節が終わってからまとめてレーティングを更新する。
    """
    columns = ["team", "elo"]
    if results_df.empty:
        return pd.DataFrame(columns=columns)

    df = results_df.copy()
    df["round_no"] = df["round"].fillna("").str.extract(r"(\d+)")
    df = df.dropna(subset=["round_no"])
    df["round_no"] = df["round_no"].astype(int)

    ratings = defaultdict(lambda: ELO_INITIAL)

    for _, round_df in df.sort_values("round_no").groupby("round_no"):
        pending_updates = defaultdict(float)
        for _, row in round_df.iterrows():
            home, away = row["home_team"], row["away_team"]
            home_goals, away_goals = row["home_goals"], row["away_goals"]

            expected_home = 1 / (1 + 10 ** (((ratings[away]) - (ratings[home] + ELO_HOME_ADVANTAGE)) / 400))

            if home_goals > away_goals:
                actual_home = 1.0
            elif home_goals < away_goals:
                actual_home = 0.0
            else:
                actual_home = 0.5

            delta = ELO_K * (actual_home - expected_home)
            pending_updates[home] += delta
            pending_updates[away] -= delta

        for team, delta in pending_updates.items():
            ratings[team] += delta

    records = [{"team": team, "elo": round(rating)} for team, rating in ratings.items()]
    return pd.DataFrame(records, columns=columns).sort_values("elo", ascending=False)


def _shrunk_rate(goals_sum: float, games_played: int, league_avg_rate: float) -> float:
    """
    観測された平均に、リーグ平均を「仮想的にPOISSON_PRIOR_GAMES試合分」混ぜて寄せる。
    試合数が少ないほどリーグ平均寄りになり、試合数が増えるほど実際の数値に近づいていく。
    """
    return (goals_sum + POISSON_PRIOR_GAMES * league_avg_rate) / (games_played + POISSON_PRIOR_GAMES)


def _poisson_params(results_df: pd.DataFrame):
    """
    リーグ平均得点と、チームごと(ホーム/アウェイ別)の攻撃力・守備力の係数を算出する。
    係数は「リーグ平均を1.0としたときの倍率」で表す(1.2なら平均より2割得点力が高い、など)。
    """
    if results_df.empty:
        return None

    league_avg_home_goals = results_df["home_goals"].mean()
    league_avg_away_goals = results_df["away_goals"].mean()
    if not league_avg_home_goals or not league_avg_away_goals:
        return None

    teams = pd.unique(results_df[["home_team", "away_team"]].values.ravel())
    team_stats = {}
    for team in teams:
        home_games = results_df[results_df["home_team"] == team]
        away_games = results_df[results_df["away_team"] == team]

        home_attack_rate = _shrunk_rate(home_games["home_goals"].sum(), len(home_games), league_avg_home_goals)
        home_defense_rate = _shrunk_rate(home_games["away_goals"].sum(), len(home_games), league_avg_away_goals)
        away_attack_rate = _shrunk_rate(away_games["away_goals"].sum(), len(away_games), league_avg_away_goals)
        away_defense_rate = _shrunk_rate(away_games["home_goals"].sum(), len(away_games), league_avg_home_goals)

        team_stats[team] = {
            "home_attack": home_attack_rate / league_avg_home_goals,
            "home_defense": home_defense_rate / league_avg_away_goals,
            "away_attack": away_attack_rate / league_avg_away_goals,
            "away_defense": away_defense_rate / league_avg_home_goals,
        }

    return {
        "league_avg_home_goals": league_avg_home_goals,
        "league_avg_away_goals": league_avg_away_goals,
        "team_stats": team_stats,
    }


def _expected_goals(params: dict, home_team: str, away_team: str):
    """指定した対戦カードの期待得点(平均何点入りそうか)を、ホーム・アウェイそれぞれ算出する。"""
    default_stats = {"home_attack": 1.0, "home_defense": 1.0, "away_attack": 1.0, "away_defense": 1.0}
    home_stats = params["team_stats"].get(home_team, default_stats)
    away_stats = params["team_stats"].get(away_team, default_stats)

    lambda_home = home_stats["home_attack"] * away_stats["away_defense"] * params["league_avg_home_goals"]
    lambda_away = away_stats["away_attack"] * home_stats["home_defense"] * params["league_avg_away_goals"]
    return lambda_home, lambda_away


def _poisson_pmf(k: int, lam: float) -> float:
    """ポアソン分布: 平均lam回起きる事象が、ちょうどk回起きる確率。"""
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _match_probabilities(lambda_home: float, lambda_away: float, max_goals: int = POISSON_MAX_GOALS):
    """
    期待得点(lambda_home, lambda_away)から、有り得るスコアの組み合わせを全て足し合わせて
    ホーム勝ち/引き分け/アウェイ勝ちの確率を計算する。
    """
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        p_h = _poisson_pmf(h, lambda_home)
        for a in range(max_goals + 1):
            p = p_h * _poisson_pmf(a, lambda_away)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    total = home_win + draw + away_win  # max_goalsを超える裾野の確率はごく僅かなので正規化して吸収する
    return home_win / total, draw / total, away_win / total


def predict_fixtures(upcoming_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    params = _poisson_params(results_df)

    rows = []
    for _, fixture in upcoming_df.iterrows():
        if params is None:
            home_win_prob, draw_prob, away_win_prob = 33.3, 33.4, 33.3  # 実績データがまだ無い場合の暫定値
        else:
            lambda_home, lambda_away = _expected_goals(params, fixture["home_team"], fixture["away_team"])
            home_win, draw, away_win = _match_probabilities(lambda_home, lambda_away)
            home_win_prob, draw_prob, away_win_prob = round(home_win * 100, 1), round(draw * 100, 1), round(away_win * 100, 1)

        rows.append({
            "fixture_id": fixture["fixture_id"],
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_win_prob": home_win_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_win_prob,
        })

    columns = ["fixture_id", "date", "home_team", "away_team", "home_win_prob", "draw_prob", "away_win_prob"]
    return pd.DataFrame(rows, columns=columns)


def main():
    for category, label in CATEGORIES.items():
        print(f"\n===== {label} =====")
        upcoming_df = pd.read_csv(_latest_file("upcoming", category))
        results_df = pd.read_csv(_latest_file("results", category))

        predictions_df = predict_fixtures(upcoming_df, results_df)
        predictions_path = os.path.join(DATA_DIR, f"predictions_{category}_{datetime.now().date()}.csv")
        predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {predictions_path}")
        print(predictions_df)

        elo_df = compute_elo_ratings(results_df)
        elo_path = os.path.join(DATA_DIR, f"elo_{category}_{datetime.now().date()}.csv")
        elo_df.to_csv(elo_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {elo_path}")
        print(elo_df.head())


if __name__ == "__main__":
    main()
